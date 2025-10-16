# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import logging
import numpy as np
import os
import shutil
import copy
import math

import torch
from PIL import Image
from datetime import datetime
from diffusers import DDPMScheduler, DDPMScheduler, FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Union

from accelerate import Accelerator

from flux import FluxDepthPipeline, FluxDepthPipelineOutput
from src.util import metric
from src.util.alignment import align_depth_least_square
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence
from src.util.flux_utils import (
    set_flux_transformer_lora,
)

from diffusers import DDPMScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.optimization import get_scheduler
from safetensors.torch import load_file

class FluxDepthMultiGPUTrainer:
    def __init__(
        self,
        accelerator: Accelerator,
        cfg: OmegaConf,
        model: FluxDepthPipeline,
        train_dataloader: DataLoader,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: FluxDepthPipeline = model
        self.device = accelerator.device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # Accelerator
        self.accelerator = accelerator
        self.is_master = self.accelerator.is_main_process

        # Encode empty text prompt
        prompt_path = self.cfg.prompt_embeds.dir
        prompt_embeds = torch.load(os.path.join(prompt_path, "prompt_embeds.pt"), weights_only=True, map_location=self.device)
        pooled_prompt_embeds = torch.load(os.path.join(prompt_path, "pooled_prompt_embeds.pt"), weights_only=True, map_location=self.device)
        text_ids = torch.load(os.path.join(prompt_path, "text_ids.pt"), weights_only=True, map_location=self.device)

        self.model.load_prompt_embeds(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids
        )

        # Trainability
        weight_dtype = torch.float32
        if self.cfg.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif self.cfg.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16        
        self.weight_dtype = weight_dtype

        self.model.vae.requires_grad_(False)
        self.model.transformer.requires_grad_(False)

        self.model.transformer = set_flux_transformer_lora(
            self.model.transformer,
            rank=self.cfg.lora.rank,
            alpha=self.cfg.lora.alpha,
            target_modules=self.cfg.lora.modules,
        )

        if self.cfg.enable_xformers_memory_efficient_attention:
            self.model.transformer.enable_xformers_memory_efficient_attention()
        
        if self.cfg.enable_gradient_checkpointing:
            self.model.transformer.enable_gradient_checkpointing()

        if self.cfg.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.transformer.parameters()))
        if self.cfg.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            
            self.optimizer = bnb.optim.AdamW8bit(trainable_parameters, lr=lr)
        else:
            self.optimizer = Adam(trainable_parameters, lr=lr)

        # LR scheduler
        # lr_func = IterExponential(
        #     total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
        #     final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
        #     warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        # )
        # self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)
        self.lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler.name,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_scheduler.warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.cfg.max_iter * self.accelerator.num_processes,
            num_cycles=self.cfg.lr_scheduler.num_cycles,
            power=self.cfg.lr_scheduler.power,
        )

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Training noise scheduler    
        self.training_scheduler = copy.deepcopy(self.model.scheduler)
        self.training_timesteps = self.training_scheduler.timesteps                 # ['1000.0', '999.0', ..., '1.0']
        self.training_sigmas = self.training_scheduler.sigmas                       # ['1.0', '0.999', ... '0.001']
        self.num_train_timesteps = self.model.scheduler.config.num_train_timesteps  # 1000

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]

        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])

        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal

        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."

        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # FLUX parameters
        self.guidance_scale = self.cfg.guidance_scale
        self.vae_scale_factor = self.model.vae_scale_factor

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

        # TODO: Path interpolation
        self.glowd_max_step = self.cfg.glowd.max_step
        self.glowd_interpolation = self.cfg.glowd.interpolation

    
    def path_interpolation(
        self,
        prior_source: torch.Tensor,
        prior_target: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        step: int,
        timestep_interval: Union[list, tuple] = [1, 0],
        c: float = 1.5,
        epsilon: float = 1e-8,
    ):
        if step >= self.glowd_max_step:
            return 0.0, source, target
        
        timestep_start, timestep_end = timestep_interval

        # quantize timestep into flux training timesteps
        t_idx = int(step / (self.glowd_max_step / self.num_train_timesteps))
        t = self.training_sigmas[t_idx].detach().cpu().item()

        if self.glowd_interpolation == "linear":
            timestep = (1 - t) * timestep_start + t * timestep_end
        elif self.glowd_interpolation == "exponential":
            timestep = (math.exp(-c*t) - math.exp(-c)) / (1 - math.exp(-c))
        elif self.glowd_interpolation == "logarithmic":
            timestep = (1.0 - math.log(1 + c * t) / math.log(1 + c))

        intermediate_source = prior_source * (1 - timestep) + source * timestep
        intermediate_target = prior_target * (1 - timestep) + target * timestep

        return timestep, intermediate_source, intermediate_target        
        

    def time_interpolation(self, source_latent, timesteps, target_latent, eps=5e-1):
        """
        Time interpolation between two latents:
                        x_1             ->          x_0
        ------------------------------------------------------------
        sample:         source (RGB)                target (Depth)
        timestep:       1000                        0
        sigma:          1.0                         0.0
        ------------------------------------------------------------
        interpolation:      x_t = t * x_1 + (1 - t) * x_0
        
        Args:
            source_latent('torch.FloatTensor'): 
                latent from the source distribution (image)
            target_latent('torch.FloatTensor'): 
                latent from the target distribution (depth)
            timesteps:
                The current timestep in the diffusion chain.
        """
        t_ = (timesteps / self.num_train_timesteps).view(-1, 1 ,1 ,1)
        interpolated_latent = t_ * source_latent + (1 - t_) * target_latent

        return interpolated_latent

    def sample_timesteps_and_sigmas(self, batch_size, device):
        """
        Sample timesteps t, s, r (t >= s >= r) or t
        
        Args:
            interval_splitting('bool'):
                If True, sample three timesteps for 'interval splitting consistency'
                If False, sample one timestep for 'boundary condition'
        """

        # Step 1: sample two timesteps
        d = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (d * len(self.training_timesteps)).long()
        
        t = self.training_timesteps[indices].to(device=device)
        sigma_t = self.training_sigmas[indices].to(device=device)
        sigma_t = sigma_t.view(batch_size, 1, 1, 1)
        
        return t, sigma_t


    def train(self, t_end=None):
        if self.is_master:
            logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0
    
        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            if self.is_master:
                logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in self.train_loader: #skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                models_to_accumulate = [self.model.transformer]
                with self.accelerator.accumulate(models_to_accumulate)
                    self.model.transformer.train()

                    # globally consistent random generators
                    if self.seed is not None:
                        local_seed = self._get_next_seed()
                        rand_num_generator = torch.Generator(device=device)
                        rand_num_generator.manual_seed(local_seed)
                    else:
                        rand_num_generator = None

                    # Get data
                    rgb = batch["rgb_norm"].to(device)                              # [b, 3, H, W]
                    depth_gt_for_latent = batch[self.gt_depth_type].to(device)

                    batch_size, _, height, width = rgb.shape

                    if self.gt_mask_type is not None:
                        num_channels_latents = self.model.transformer.config.in_channels // 4
                        valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                        invalid_mask = ~valid_mask_for_latent
                        valid_mask_down = ~torch.max_pool2d(
                            invalid_mask.float(), 8, 8
                        ).bool()
                        valid_mask_down = valid_mask_down.repeat((1, num_channels_latents, 1, 1))

                    with torch.no_grad():
                        # Encode image
                        rgb_latent = self.encode_rgb(rgb)

                        # Encode GT depth
                        gt_target_latent = self.encode_depth(
                            depth_gt_for_latent
                        )  # [B, 4, h, w]

                    # Text embedding
                    prompt_embeds = self.empty_prompt_embeds.to(device).repeat(batch_size, 1, 1)
                    pooled_prompt_embeds = self.empty_pooled_prompt_embeds.to(device).repeat(batch_size, 1)
                    text_ids = self.text_ids.to(device)

                    # guidance vector
                    guidance = torch.full([1], self.guidance_scale, device=device, dtype=torch.float32)
                    guidance = guidance.expand(batch_size)
                
                    # Sample timesteps & interpolate path
                    timestep, interp_rgb_latent, interp_gt_target_latent = self.path_interpolation(
                        prior_source=torch.randn_like(rgb_latent),
                        prior_target=rgb_latent,
                        source=rgb_latent,
                        target=gt_target_latent,
                        step=self.effective_iter,
                    )
                    t = torch.tensor(timestep, device=device, dtype=torch.float32)

                # t = torch.tensor(0., device=device, dtype=torch.float32)
                # t = t.expand(batch_size)

                    # prepare latents
                    packed_rgb_latent, rgb_latent_image_ids = self.prepare(latents=interp_rgb_latent)

                    # One forward pass
                    model_pred = self.model.transformer(
                        hidden_states=packed_rgb_latent,
                        timestep=t / 1000,      # t in [0, 1]
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=rgb_latent_image_ids,
                        return_dict=False
                    )[0]

                    model_pred = self.model._unpack_latents(
                        model_pred,
                        height=height,
                        width=width,
                        vae_scale_factor=self.vae_scale_factor,
                    )

                    target = interp_rgb_latent - interp_gt_target_latent

                    # Loss
                    if self.gt_mask_type is not None:
                        latent_loss = self.loss(
                            model_pred[valid_mask_down].float(),
                            target[valid_mask_down].float(),
                        )
                    else:
                        latent_loss = self.loss(model_pred.float(), target.float())

                    loss = latent_loss.mean()
                    self.train_metrics.update("loss", loss.item())

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    self.n_batch_in_epoch += 1
                    # Practical batch end

                # Perform optimization step
                if self.accelerator.sync_gradients:
                    self.effective_iter += 1
                    if self.is_master:
                        # Log to tensorboard
                        accumulated_loss = self.train_metrics.result()["loss"]
                        tb_logger.log_dict(
                            {
                                f"train/{k}": v
                                for k, v in self.train_metrics.result().items()
                            },
                            global_step=self.effective_iter,
                        )
                        tb_logger.writer.add_scalar(
                            "lr",
                            self.lr_scheduler.get_last_lr()[0],
                            global_step=self.effective_iter,
                        )
                        tb_logger.writer.add_scalar(
                            "n_batch_in_epoch",
                            self.n_batch_in_epoch,
                            global_step=self.effective_iter,
                        )
                        logging.info(
                            f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                        )
                        self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()
                    self.accelerator.wait_for_everyone()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter and self.is_master:
                        self.wait_for_everyone()
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        self.accelerator.end_training()
                        logging.info("Training ended.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def encode_rgb(self, image_in):
        assert len(image_in.shape) == 4 and image_in.shape[1] == 3
        latent = self.model.encode_rgb(image_in)
        return latent

    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.model.encode_rgb(stacked)
        return depth_latent

    def prepare_latents(self, images=None, latents=None):
        assert images is None and latents is None, "either image or latents should be provided!"
        if latents is None:
            latents = self.encode_rgb(images)

        batch_size, num_channels_latents, height, width = latents.shape
        packed_latents = self.model._pack_latents(
            latents,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
        )

        latent_image_ids = self.model._prepare_latent_image_ids(
            batch_size=batch_size,
            height=height // 2,
            width=width // 2,
            device=latents.device,
            dtype=latents.dtype
        )

        return packed_latents, latent_image_ids

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1).repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dict = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dict}"
            )
            tb_logger.log_dict(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dict.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dict_to_text(
                val_metrics=val_metric_dict,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dict[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # Save a checkpoint
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            # Read input image
            rgb_int = batch["rgb_int"]  # [B, 3, H, W]
            # GT depth
            depth_raw_ts = batch["depth_raw_linear"].squeeze()
            depth_raw = depth_raw_ts.numpy()
            depth_raw_ts = depth_raw_ts.to(self.device)
            valid_mask_ts = batch["valid_mask_raw"].squeeze()
            valid_mask = valid_mask_ts.numpy()
            valid_mask_ts = valid_mask_ts.to(self.device)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict depth for different inference_steps
            pipe_out: FluxDepthPipelineOutput = self.model(
                rgb_int,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                color_map=None,
                guidance_scale=self.guidance_scale,
            )

            depth_pred: np.ndarray = pipe_out.depth_np

            if "least_square" == self.cfg.eval.alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=depth_raw,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=self.cfg.eval.align_max_res,
                )
            else:
                raise RuntimeError(f"Unknown alignment type: {self.cfg.eval.alignment}")

            # Clip to dataset min max
            depth_pred = np.clip(
                depth_pred,
                a_min=data_loader.dataset.min_depth,
                a_max=data_loader.dataset.max_depth,
            )

            # clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # Evaluate
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)

            # Save as 16-bit uint png
            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")

                # save depth
                save_to_dir_depth = os.path.join(save_to_dir, "depth")
                os.makedirs(save_to_dir_depth, exist_ok=True)

                png_save_path = os.path.join(save_to_dir_depth, f"{img_name}.png")
                depth_to_save = (pipe_out.depth_np * 65535.0).astype(np.uint16)
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")
            
                # save image            
                save_to_dir_image = os.path.join(save_to_dir, "image")
                os.makedirs(save_to_dir_image, exist_ok=True)

                png_save_path = os.path.join(save_to_dir_image, f"{img_name}.png")
                image_to_save = rgb_int.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                Image.fromarray(image_to_save).save(png_save_path)

        return metric_tracker.result()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save Transformer
        transformer_path = os.path.join(ckpt_dir, "transformer")
        self.model.transformer.save_pretrained(transformer_path, safe_serialization=True)
        logging.info(f"Transformer is saved to: {transformer_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load Transformer
        _model_path = os.path.join(ckpt_path, "transformer", "diffusion_pytorch_model.safetensors")
        self.model.transformer.load_state_dict(
            load_file(_model_path, device='cpu')
        )
        self.model.transformer.to(self.device)
        logging.info(f"Transformer parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
