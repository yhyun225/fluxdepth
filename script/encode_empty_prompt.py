import os
import torch

from diffusers import FluxPipeline

pretrained_model_name_or_path = "/home/yhyun225/fluxdepth/checkpoint/FLUX.1-dev"
prompt_embed_dir = "/home/yhyun225/fluxdepth/prompt_embeds"

os.makedirs(prompt_embed_dir, exist_ok=True)

text_encoding_pipeline = FluxPipeline.from_pretrained(
    pretrained_model_name_or_path, torch_dtype=torch.bfloat16,
) #.to('cuda')

breakpoint()

empty_prompt = ""
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
        prompt=empty_prompt, prompt_2=None,
    )

torch.save(prompt_embeds, os.path.join(prompt_embed_dir, "prompt_embeds.pt"))
torch.save(pooled_prompt_embeds, os.path.join(prompt_embed_dir, "pooled_prompt_embeds.pt"))
torch.save(text_ids, os.path.join(prompt_embed_dir, "text_ids.pt"))

