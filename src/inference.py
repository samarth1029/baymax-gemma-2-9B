import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaymaxInference:
    def __init__(self, model_name="samarth1029/Gemma-2-9b-baymax", device="cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    def _load_model(self):
        """Load the pre-trained model from Hugging Face."""
        print("Loading model...")
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True
        ).to(self.device)

    def _load_tokenizer(self):
        """Load the tokenizer associated with the model."""
        print("Loading tokenizer...")
        return AutoTokenizer.from_pretrained(self.model_name)

    def format_prompt(self, messages):
        """Apply a chat template formatting to the input messages for inference."""
        print("Formatting prompt...")
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def tokenize_input(self, prompt):
        """Tokenize the input prompt and prepare tensors for model generation."""
        print("Tokenizing input...")
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True, truncation=True
        ).to(self.device)

    def generate_response(self, inputs, max_new_tokens=150, num_return_sequences=1):
        """Generate a response from the model based on tokenized inputs."""
        print("Generating response...")
        return self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences
        )

    def decode_response(self, outputs):
        """Decode the generated output tokens to a readable string."""
        print("Decoding response...")
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def infer(self, messages):
        """Full pipeline for inference, from formatting to generating and decoding the response."""
        prompt = self.format_prompt(messages)
        inputs = self.tokenize_input(prompt)
        outputs = self.generate_response(inputs)
        return self.decode_response(outputs)


# Sample usage
if __name__ == "__main__":
    baymax_inference = BaymaxInference()
    queries = [
        {"role": "user",
         "content": "Hello, I am in the middle of a severe anxiety/panic attack. Could you help me?"}
    ]
    response = baymax_inference.infer(queries)
    print("Generated Response:", response)
