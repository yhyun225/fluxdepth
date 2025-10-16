export BASE_DATA_DIR='/scratch/yhyun225/marigold_dataset/train'  # directory of training data
export BASE_CKPT_DIR='checkpoint'  # directory of pretrained checkpoint

CUDA_VISIBLE_DEVICES=3 python script/depth/train.py \
    --config config/train_flux_depth.yaml \
    --output_dir output_flux_depth
    # --no_wandb