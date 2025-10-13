from peft import LoraConfig, PeftModel

def set_vae_encoder_lora(vae_encoder, rank):
    target_modules = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]

    vae_encoder_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    vae_encoder = PeftModel(
        model=vae_encoder,
        peft_config=vae_encoder_lora_config,
        adapter_name="vae_encoder_lora_adapter",
    )
    vae_encoder.print_trainable_parameters()
    return vae_encoder

def get_transformer_lora_config(rank, alpha=None, dropout=0.0):
    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank if alpha is None else alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    
    return transformer_lora_config