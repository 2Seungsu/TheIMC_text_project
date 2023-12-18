import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class MySummarizer:
    def __init__(self, model, weights_path):
        # tokenizer, model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)

        # Load weights and apply model
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()

    def generate_summary(self, example_text, max_length=40):
        inputs = self.tokenizer(example_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
        # Added code with token_type_ids error when progressed in the kaggle
        inputs.pop("token_type_ids", None)

        with torch.no_grad(): # For verification, no_grad reduces computational speed and memory usage
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
            )

        # Decode the generated summary
        generated_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_summary
