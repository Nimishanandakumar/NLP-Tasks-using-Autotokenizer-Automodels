from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Input text
text = "I love using Hugging Face Transformers!"

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

# Print the result
print(f"Predicted class: {predicted_class}")  # 1 for positive, 0 for negative