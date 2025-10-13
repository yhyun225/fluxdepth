import logging
import numpy as np
import torch
import PIL
from PIL import Image
from typing import List, Union
from dataclasses import dataclass

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models import FluxTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Union

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

logger = logging.get_logger(__name__)

@dataclass
class FluxDepthPipelineOutput(BaseOutput):
    """
    Output class for Flux image generation pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `torch.Tensor` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array or torch tensor of shape `(batch_size,
            height, width, num_channels)`. PIL images or numpy array present the denoised images of the diffusion
            pipeline. Torch tensors can represent either the denoised images or the intermediate latents ready to be
            passed to the decoder.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]

class FluxDepthPipeline(DiffusionPipeline):
    
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        prediction_type: str = 'depth',
        empty_prompt_embeds: Optional[torch.Tensor] = None,
        emtpy_pooled_prompt_ebmeds: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            prediction_type=prediction_type,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

        self.empty_prompt_embeds = empty_prompt_embeds
        self.empty_pooled_prompt_embeds = emtpy_pooled_prompt_ebmeds
        self.text_ids = text_ids
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids
    
    def load_prompt_embeds(
        self,
        empty_prompt_embeds,
        empty_pooled_prompt_embeds,
        text_ids,
    ):
        self.empty_prompt_embeds = empty_prompt_embeds
        self.empty_pooled_prompt_embeds = empty_pooled_prompt_embeds
        self.text_ids = text_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents
    
    def encode_rgb(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB images into latents.

        Args:
            images ('torch.Tensor'):    Input RGB images to be encoded.
            normalized (bool):          Whether the image is prenormalized (torch values in [-1, 1]).
            
        Returns:
            latents ('torch.Tensor'):   Image latents.
        """
        latents = self.vae.encode(images).latent_dist.mode()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        return latents
    
    def decode_depth(self, latents:torch.Tensor) -> torch.Tensor:
        """
        Decode depth latents into depth map.

        Args:
            latents ('torch.Tensor'):   Depth latents to be decoded.

        Returns:
            depths ('torch.Tensor'):    1-dimensional depth map
        """
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        depths = self.vae.decode(latents, return_dict=False)[0]
        depths = depths.mean(dim=1, keepdim=True)   # [b, 1, h, w]

        return depths

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        processing_res: Optional[int] = 768,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        guidance_scale: float = 1.0,
    ) -> FluxDepthPipelineOutput:
        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        batch_size = rgb.shape[0]

        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm = rgb / 255.0 * 2.0 - 1.0
        rgb_norm = rgb_norm.to(dtype=self.dtype, device=self.device)

        # Encode image
        rgb_latents = self.encode_rgb(rgb_norm)
        _, c, h, w = rgb_latents.shape
        rgb_latents = self._pack_latents(
            rgb_latents,
            batch_size=batch_size,
            num_channels_latents=c,
            height=h,
            width=w,
        )
        rgb_latents_image_ids = self._prepare_latent_image_ids(
            batch_size,
            h // 2,
            w //2,
            device=self.device,
            dtype=self.dtype
        )

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=self.device, dtype=self.dtype)
            guidance = guidance.expand(batch_size)
        else:
            guidance = None
        
        timestep = torch.tensor([1.0], device=self.device)

        # Forward once!
        pred_depth_latents = self.transformer(
            hidden_states=rgb_latents,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=self.empty_pooled_prompt_embeds,
            encoder_hidden_states=self.empty_prompt_embeds,
            txt_ids=self.text_ids,
            img_ids=rgb_latents_image_ids,
            return_dict=False,
        )[0]

        depth_latents = self._unpack_latents(
            pred_depth_latents,
            height=h*self.vae_scale_factor,
            width=w*self.vae_scale_factor,
            vae_scale_factor=self.vae_scale_factor,
        )
        depth = self.decode_depth(pred_depth_latents)
        depth = torch.clip(depth, -1.0, 1.0)

        # denormalize the depths: [-1, 1] -> [0, 1]
        depth = (depth + 1.0) / 2.0

        if match_input_res:
            depth = resize(
                depth,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )
        depth = depth.squeeze()
        depth = depth.cpu().numpy()
        depth = depth.clip(0, 1)

        # Colorize
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                depth, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return FluxDepthPipelineOutput(
            depth_np=depth,
            depth_colored=depth_colored_img,
            uncertainty=None
        )


