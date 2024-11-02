from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model


class ModelLoader:
    def __init__(self, model_name, lora_config=None):
        self.model_name = model_name
        self.lora_config = lora_config

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    def prepare_for_training(self, model):
        model = prepare_model_for_kbit_training(model)
        if self.lora_config:
            model = get_peft_model(model, self.lora_config)
        return model
