tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi") 
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

def translate_text(text): 
  inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True) 
  translated_tokens = model.generate(**inputs, max_length=256) 
  translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0] 
  return translated_text

input_text = "Hello, how are you?" 
translated_output = translate_text(input_text) 
print(translated_output)