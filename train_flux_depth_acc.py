#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import sys
import argparse
import logging
import math
import os
from pathlib import Path
from typing import Callable
from omegaconf import OmegaConf
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    ProjectConfiguration,
    set_seed,
)
from tqdm.auto import tqdm
from torchvision.utils import save_image
import diffusers
from diffusers import (
    AutoencoderKL, FluxPipeline, FluxTransformer2DModel
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    free_memory,
)
from diffusers.utils.torch_utils import is_compiled_module
import torch.nn.functional as F
import warnings

from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, PeftModel
import copy

from flux import FluxDepthPipeline

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def _prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, _, channels = latents.shape

    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def get_flux_setting_timesteps(n=999):
    return get_schedule(
        n,
        (1024 // 8) * (1024 // 8) // 4,
        shift=True,
    )

def set_flux_transformer_lora(flux_transformer, rank):
    target_modules = [
        "x_embedder",
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    flux_transformer = PeftModel(flux_transformer, transformer_lora_config, adapter_name="flux_adapter")
    flux_transformer.print_trainable_parameters()
    return flux_transformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducibility of experiments.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_flux_depth_acc.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume_run",
        action="store",
        default=None,
        help="Path of checkpoint to be resumed. If given, will ignore --config, and checkpoint in the config.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save checkpoints."
    )
    parser.add_argument(
        "--logging_dir", type=str, default="logs", help="Directory to save logs."
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Run without Weights and Biases logging.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Mixed precision.",
    )
    parser.add_argument(
        "--base_data_dir", type=str, default=None, help="Base path to the datasets."
    )
    parser.add_argument(
        "--base_ckpt_dir",
        type=str,
        default=None,
        help="Base path to the pretrained checkpoints.",
    )
    args = parser.parse_args()

    return args.config


def main():
    args = parse_args()
    resume_run = args.resume_run
    output_dir = args.output_dir
    logging_dir = Path(args.output_dir, args.logging_dir)
    base_data_dir = (
        args.base_data_dir
        if args.base_data_dir is not None
        else os.environ["BASE_DATA_DIR"]
    )
    base_ckpt_dir = (
        args.base_ckpt_dir
        if args.base_ckpt_dir is not None
        else os.environ["BASE_CKPT_DIR"]
    )
    cfg = recursive_load_config(args.config)
    
    # Accelerate configs
    num_gpus = torch.cuda.device_count()
    gradient_accumulation_steps = cfg.dataloader.eff_bs / (cfg.dataloader.max_train_batch_size * num_gpus)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            OmegaConf.save(args, os.path.join(args.output_dir, "config.yaml"))

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # prompt embeds
    prompt_path = cfg.prompt_embeds.dir
    prompt_embeds = torch.load(os.path.join(prompt_path, "prompt_embeds.pt"), weights_only=True, map_location=accelerator.device)
    pooled_prompt_embeds = torch.load(os.path.join(prompt_path, "pooled_prompt_embeds.pt"), weights_only=True, map_location=accelerator.device)
    text_ids = torch.load(os.path.join(prompt_path, "text_ids.pt"), weights_only=True, map_location=accelerator.device)

    # flux_transformer
    transformer = FluxTransformer2DModel.from_pretrained(args.flux_path, subfolder="transformer", torch_dtype=weight_dtype)    
    transformer.requires_grad_(False)
    transformer = set_flux_transformer_lora(transformer, args.flux_transformer_lora_rank)
    
    # xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available, please install it by running `pip install xformers`"
            )
    
    # gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        
    # Flux Depth Pipeline
    _pipeline_kwargs = cfg.pipeline.kwargs if cfg.pipeline.kwargs is not None else {}
    model = FluxDepthPipeline.from_pretrained(
        os.path.join(base_ckpt_dir, cfg.model.pretrained_path), transformer=transformer, **_pipeline_kwargs,
    )
    model = model.to(accelerator.device)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(
        f"Total flux_transformer training parameters: {sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]):,}"
    )
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer_sr = optimizer_class(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset
    loader_seed = cfg.dataloader.seed
    if loader_seed is None:
        loader_generator = None
    else:
        loader_generator = torch.Generator().manual_seed(loader_seed)

    # Training dataset
    depth_transform: DepthNormalizerBase = get_depth_normalizer(
        cfg_normalizer=cfg.depth_normalization
    )
    train_dataset: Union[BaseDepthDataset, List[BaseDepthDataset]] = get_dataset(
        cfg_data.train,
        base_data_dir=base_data_dir,
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
    )
    logging.debug("Augmentation: ", cfg.augmentation)
    if "mixed" == cfg_data.train.name:
        dataset_ls = train_dataset
        assert len(cfg_data.train.prob_ls) == len(
            dataset_ls
        ), "Lengths don't match: `prob_ls` and `dataset_list`"
        concat_dataset = ConcatDataset(dataset_ls)
        mixed_sampler = MixedBatchSampler(
            src_dataset_ls=dataset_ls,
            batch_size=cfg.dataloader.max_train_batch_size,
            drop_last=True,
            prob=cfg_data.train.prob_ls,
            shuffle=True,
            generator=loader_generator,
        )
        train_loader = DataLoader(
            concat_dataset,
            batch_sampler=mixed_sampler,
            num_workers=cfg.dataloader.num_workers,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.dataloader.max_train_batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=True,
            generator=loader_generator,
        )
    # Validation dataset
    val_loaders: List[DataLoader] = []
    for _val_dict in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dict,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        val_loaders.append(_val_loader)

    # Visualization dataset
    vis_loaders: List[DataLoader] = []
    for _vis_dict in cfg_data.vis:
        _vis_dataset = get_dataset(
            _vis_dict,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL,
        )
        _vis_loader = DataLoader(
            dataset=_vis_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
        )
        vis_loaders.append(_vis_loader)

    train_dataset = PairedDataset(args.dataset_txt_or_dir_paths, args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_sr = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_sr,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    (
        lora_vae,
        flux_transformer,
        net_disc,
        optimizer_sr,
        optimizer_disc,
        train_dataloader,
        lr_scheduler_sr,
        lr_scheduler_disc,
    ) = accelerator.prepare(
        lora_vae,
        flux_transformer,
        net_disc,
        optimizer_sr,
        optimizer_disc,
        train_dataloader,
        lr_scheduler_sr,
        lr_scheduler_disc,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info(f"***** Start training {args.model} *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            # accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            # global_step = int(path.split("-")[1])

            # initial_global_step = global_step
            # first_epoch = global_step // num_update_steps_per_epoch
            # TODO
            pass
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # latent image ids for RoPE.
    latent_image_ids = _prepare_latent_image_ids(
        (args.resolution // 8) // 2,
        (args.resolution // 8) // 2,
        accelerator.device,
        weight_dtype,
    )
    vae_scale_factor = 2 ** (len(fixed_vae.config.block_out_channels) - 1)

    def one_mid_timestep_pred(lq_latent):
        bsz, c, h, w = lq_latent.shape
        guidance_vec = torch.full(
            (bsz,),
            1.0,
            device=lq_latent.device,
            dtype=weight_dtype,
        )
       
        lq_latent = _pack_latents(
            lq_latent,
            batch_size=bsz,
            num_channels_latents=c,
            height=h,
            width=w,
        )
        model_pred = flux_transformer(
            hidden_states=lq_latent,
            timestep=torch.tensor([sigma_t], device=lq_latent.device), 
            guidance=guidance_vec,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        lq_latent = lq_latent - sigma_t * model_pred  
        lq_latent = _unpack_latents(
            lq_latent,
            height=h * vae_scale_factor,
            width=w * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )
        lq_latent = (lq_latent / fixed_vae.config.scaling_factor) + fixed_vae.config.shift_factor
        pred_img = fixed_vae.decode(lq_latent.to(fixed_vae.dtype), return_dict=False)[0]
        return pred_img
    
    for epoch in range(first_epoch, args.num_train_epochs):
        lora_vae.train()
        flux_transformer.train()
        net_disc.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(*[lora_vae, flux_transformer, net_disc]):
                # Prepare data
                lq_img, hq_img = batch
                lq_img = lq_img.to(accelerator.device)
                hq_img = hq_img.to(accelerator.device)
                with torch.no_grad():
                    hq_latent = encode_images(hq_img, fixed_vae, weight_dtype)
                    noise = torch.randn_like(hq_latent)
                    pretrained_noisy_latent = (1 - sigma_t) * hq_latent + sigma_t * noise  

                lq_latent = encode_images(lq_img, unwrap_model(lora_vae))

                # LRR Loss: Latent Representation Refinement Loss
                loss_LRR = F.mse_loss(pretrained_noisy_latent, lq_latent, reduction="mean") * args.lambda_LRR
                
                # Onestep prediction at mid-timestep
                pred_img = one_mid_timestep_pred(lq_latent)

                # DINOv3-ConvNext DISTS Loss
                loss_Dv3D = net_dv3d(pred_img, hq_img).mean() * args.lambda_Dv3D

                # L1 Loss
                loss_L1 = F.l1_loss(pred_img, hq_img, reduction="mean") * args.lambda_L1

                # Generator Loss (SD/FLUX)
                loss_G = net_disc(pred_img, for_G=True).mean() * args.lambda_GAN
                
                total_G_loss = loss_LRR + loss_Dv3D + loss_G + loss_L1

                accelerator.backward(total_G_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(sr_opt, args.max_grad_norm)

                optimizer_sr.step()
                lr_scheduler_sr.step()
                optimizer_sr.zero_grad()
                
                fake_img = pred_img.detach()
                # Fake images
                loss_D_fake = net_disc(fake_img, for_real=False).mean() * args.lambda_GAN 
                # Real images
                loss_D_real = net_disc(hq_img, for_real=True).mean() * args.lambda_GAN 
          
                total_D_loss = loss_D_real + loss_D_fake 

                accelerator.backward(total_D_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(disc_opt, args.max_grad_norm)

                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if (
                    accelerator.is_main_process
                    and global_step % args.save_img_steps == 0
                ):
                    img_path = os.path.join(args.output_dir, f"img-{global_step}.png")
                    save_imgs = (torch.stack([lq_img[0], pred_img[0], hq_img[0]], dim=0) + 1) / 2
                    save_image(save_imgs.detach(), img_path)
                    logger.info(f"img-{global_step}.png saved!")

                progress_bar.update(1)
                global_step += 1

                if (
                    accelerator.is_main_process
                    or accelerator.distributed_type == DistributedType.DEEPSPEED
                ):
                    if global_step % args.checkpointing_steps == 0:
                        weight_path = os.path.join(
                            args.output_dir, f"weight-{global_step}"
                        )
                        os.makedirs(weight_path, exist_ok=True)
                        unwrap_model(flux_transformer).save_pretrained(weight_path)
                        unwrap_model(lora_vae).encoder.save_pretrained(weight_path)
                        logger.info(f"Saved weight to {weight_path}")

            logs = {
                "loss_LRR": loss_LRR.detach().item(),
                "loss_D_fake": loss_D_fake.detach().item(),
                "loss_D_real": loss_D_real.detach().item(),
                "loss_Dv3D": loss_Dv3D.detach().item(),
                "loss_L1": loss_L1.detach().item(),
                "lr": lr_scheduler_sr.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        weight_path = os.path.join(args.output_dir, f"weight-{global_step}")
        os.makedirs(weight_path, exist_ok=True)
        unwrap_model(flux_transformer).save_pretrained(weight_path)
        unwrap_model(lora_vae).encoder.save_pretrained(weight_path)
        logger.info(f"Saved weight to {weight_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()