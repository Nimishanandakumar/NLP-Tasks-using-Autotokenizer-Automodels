from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input prompt
prompt = "Once upon a time"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=50)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(f"Generated text: {generated_text}")