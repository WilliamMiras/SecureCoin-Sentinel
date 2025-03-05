import torch
from transformers import RobertaTokenizer
from models.codebert import CodeBERTClassifier

# Load model and tokenizer
model = CodeBERTClassifier.load("../models/my_codebert_goodbad_model")
tokenizer = RobertaTokenizer.from_pretrained("../models/my_codebert_goodbad_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# New code to classify
new_code = ["def subtract(a, b): return a - b"]
encodings = tokenizer(new_code, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

# Inference
with torch.no_grad():
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    outputs = model(input_ids, attention_mask)
    prediction = torch.argmax(outputs, dim=-1).item()
    print(f"Prediction: {'good' if prediction == 0 else 'bad'}")