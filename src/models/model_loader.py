import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import setup_chat_format


class ModelLoader:
    def __init__(self, base_model_name, lora_config):
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.lora_config = lora_config

    def load_model(self, torch_dtype, attn_implementation):
        """Load the base model with specified quantization and attention implementation."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implementation
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)

    def find_all_linear_names(self):
        """Identify all target linear layers for LoRA."""
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:  # remove 'lm_head' if needed for certain configurations
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def setup_peft_model(self):
        """Configure the model with LoRA and setup chat formatting."""
        # Update the lora_config with target modules
        modules = self.find_all_linear_names()
        self.lora_config.target_modules = modules

        # Setup the model with PEFT
        self.model = get_peft_model(self.model, self.lora_config)
        self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)
