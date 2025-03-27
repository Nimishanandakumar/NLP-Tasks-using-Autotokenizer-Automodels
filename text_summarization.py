from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "facebook/bart-large-cnn"  # A model fine-tuned for summarization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input text
text = (
    "Hugging Face is creating a tool that democratizes AI. "
    "It provides easy access to state-of-the-art models and datasets, "
    "making it easier for developers and researchers to build and deploy machine learning applications."
)

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

# Run inference
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=50, num_beams=4, early_stopping=True)

# Decode the generated summary
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the summary
print(f"Summary: {summary}")