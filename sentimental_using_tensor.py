from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import pipeline
import torch.nn.functional as F

# Load the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model_name,tokenizer=tokenizer)

X_train = ["I've been waiting for a hugging face course my whole life.", "Python is great"]

res = classifier(X_train)
print(res)

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim = 1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)
    print(labels)