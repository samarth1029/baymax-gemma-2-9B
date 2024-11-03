from data.data_loader import DataLoader
from models.model_loader import ModelLoader
from utils.auth import Authenticator
from utils.config import get_lora_config, get_training_arguments
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


class Baymax:
    def __init__(self, base_model="google/gemma-2-2b-it", new_model="Gemma-2-2b-baymax"):
        self.base_model = base_model
        self.new_model = new_model
        self.authenticator = Authenticator()
        self.model_loader = None
        self.dataset = None
        self.trainer = None
        self.DATASET_PATH = "lavita/ChatDoctor-HealthCareMagic-100k"

    def authenticate(self):
        """Authenticate with Hugging Face and WandB using tokens from .env file."""
        self.authenticator.authenticate()

    def setup_model(self):
        """Initialize model loader, load model, and setup LoRA configuration."""
        attn_implementation = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
        torch_dtype = torch.bfloat16 if attn_implementation == "flash_attention_2" else torch.float16

        # Initialize model loader with base model and LoRA configuration
        self.model_loader = ModelLoader(self.base_model, get_lora_config())
        self.model_loader.load_model(torch_dtype, attn_implementation)
        self.model_loader.setup_peft_model()

    def load_data(self):
        """Load and preprocess the dataset."""
        data_loader = DataLoader(self.DATASET_PATH, self.model_loader.tokenizer)
        self.dataset = data_loader.load_data()
        self.dataset = data_loader.preprocess(self.dataset)

    def setup_trainer(self):
        """Initialize the SFTTrainer with model, dataset, tokenizer, and training arguments."""
        training_args = get_training_arguments(self.new_model)
        self.trainer = SFTTrainer(
            model=self.model_loader.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            peft_config=self.model_loader.lora_config,
            max_seq_length=512,
            dataset_text_field="text",
            tokenizer=self.model_loader.tokenizer,
            args=training_args,
            packing=False,
        )

    def train_model(self):
        """Train the model using the trainer."""
        self.trainer.train()

    def save_and_push_model(self):
        """Save the trained model and push it to the Hugging Face hub."""
        # Save the fine-tuned model
        self.trainer.model.save_pretrained(self.new_model)
        self.trainer.model.push_to_hub(self.new_model, use_temp_dir=False)

    def merge_and_save_final_model(self):
        """Merge LoRA weights with the base model and save the final model."""
        # Reload base model and merge LoRA weights
        base_model_reload = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model_reload, self.new_model)
        model = model.merge_and_unload()

        # Save and push the merged model
        model.save_pretrained(self.new_model)
        self.model_loader.tokenizer.save_pretrained(self.new_model)
        model.push_to_hub(self.new_model, use_temp_dir=False)
        self.model_loader.tokenizer.push_to_hub(self.new_model, use_temp_dir=False)

    def run(self):
        """Run the complete Baymax pipeline: authentication, setup, training, and saving."""
        print("Starting Baymax pipeline...")
        self.authenticate()
        print("Authentication successful.")

        self.setup_model()
        print("Model setup complete.")

        self.load_data()
        print("Data loading and preprocessing complete.")

        self.setup_trainer()
        print("Trainer setup complete.")

        self.train_model()
        print("Training complete.")

        self.save_and_push_model()
        print("Fine-tuned model saved and pushed to hub.")

        self.merge_and_save_final_model()
        print("Merged model saved and pushed to hub. Baymax pipeline completed.")


if __name__ == "__main__":
    baymax = Baymax()
    baymax.run()
