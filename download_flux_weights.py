import os
from huggingface_hub import snapshot_download

# # Download sd2.1-base: 512x512
# download_dir = 'checkpoint/stable-diffusion-2-1-base'
# os.makedirs(download_dir, exist_ok=True)

# snapshot_download(repo_id="stabilityai/stable-diffusion-2-1-base", repo_type="model", local_dir=download_dir)

# Download sd2.1: 768 x 768
download_dir = 'checkpoint/FLUX.1-dev'
os.makedirs(download_dir, exist_ok=True)

snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", repo_type="model", local_dir=download_dir)


# Download sd3
# download_dir = "checkpoint/stable-diffusion-3-medium"
# os.makedirs(download_dir, exist_ok=True)

# snapshot_download(repo_id="stabilityai/stable-diffusion-3-medium-diffusers", repo_type="model", local_dir=download_dir)