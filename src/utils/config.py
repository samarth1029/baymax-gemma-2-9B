from peft import LoraConfig


def get_lora_config():
    return LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
