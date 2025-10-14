from peft import PeftModel, LoraConfig

def set_flux_transformer_lora(transformer, rank, alpha, target_modules: list = None):
    # NOTE: No add_adapter() method implemented with FLUX transformer.
    # We explicitly hand over lora-adapted transformer in here.
    if target_modules is None:
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
        lora_alpha=alpha,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    lora_transformer = PeftModel(
        transformer, transformer_lora_config, adapter_name="flux_lora_adatper"
    )
    lora_transformer.print_trainable_parameters()
    
    return lora_transformer