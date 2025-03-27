from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Input text
text = "Hi, my name is Ganesh Lokare. I am from Pune."

# Tokenize the input
inputs = tokenizer(text, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
logits = outputs.logits
predicted_classes = torch.argmax(logits, dim=2)

# Decode the tokens and print entities
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, pred_class in zip(tokens, predicted_classes[0]):
    if pred_class != 0:  # 0 is usually the label for 'O' (no entity)
        print(f"Token: {token}, Entity: {model.config.id2label[pred_class.item()]}")