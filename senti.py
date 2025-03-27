from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import pipeline

# Load the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model_name,tokenizer=tokenizer)

res = classifier("I've been waiting for a hugging face course my whole life.")

print(res)

sequence = "USing a transformer network is simple"
res = tokenizer(sequence)
print(res)
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)