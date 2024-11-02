from huggingface_hub import login
import wandb
from dotenv import load_dotenv
import os


class Authenticator:
    def __init__(self):
        load_dotenv()
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        self.wandb_token = os.getenv("WANDB_TOKEN")

    def authenticate(self):
        if not self.hf_token or not self.wandb_token:
            raise ValueError("HUGGING_FACE_TOKEN or WANDB_TOKEN is not set in the .env file.")

        login(token=self.hf_token)
        wandb.login(key=self.wandb_token)
        print("Authenticated with Hugging Face and WandB")
