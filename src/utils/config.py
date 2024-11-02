from peft import LoraConfig
from transformers import TrainingArguments


def get_lora_config():
    """Return LoRA configuration for model fine-tuning."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None
    )


def get_training_arguments(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="wandb"
    )
