export BASE_DATA_DIR='/data1/yhyun225/dataset/marigold_dataset/train'  # directory of training data
export BASE_CKPT_DIR='checkpoint'  # directory of pretrained checkpoint

CUDA_VISIBLE_DEVICES=0 python script/depth/train.py \
    --config config/train_flux_depth.yaml \
    --output_dir output_flux_depth
    # --no_wandb