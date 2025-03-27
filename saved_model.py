from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import pipeline
import torch.nn.functional as F


save_directory ="saved"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

tok = Autotokenizer.from_pretrained(save_directory)
mod = AutoModelForSequenceClassificayion.from_pretrained(save_directory)