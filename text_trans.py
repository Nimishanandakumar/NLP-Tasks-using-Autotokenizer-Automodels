from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input text
input_text = "Hello, how are you?"

# Tokenization
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Generate translation
outputs = model.generate(**inputs, max_length=40, num_beams=4, temperature=1.0)

# Decode the generated tokens
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)