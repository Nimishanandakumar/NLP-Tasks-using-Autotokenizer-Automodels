from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Input context and question
context = "Hugging Face is a company that provides tools for natural language processing."
question = "What does Hugging Face provide?"

# Tokenize the input
inputs = tokenizer(question, context, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the start and end logits
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Get the predicted start and end positions
start_index = torch.argmax(start_logits)
end_index = torch.argmax(end