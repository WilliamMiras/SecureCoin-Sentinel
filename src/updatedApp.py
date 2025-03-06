#Updated API to support both CodeBERT and FNN predictions.

from flask import Flask, request, jsonify
from transformers import RobertaTokenizer
from models.codebert import CodeBERTClassifier
from models.fnn import ScamDetectionFNN
from dataset import load_and_preprocess_fnn_data
import torch

app = Flask(__name__)

# Load CodeBERT
codebert = CodeBERTClassifier.load("models/codeBERT_smartContractAnalysis")
tokenizer = RobertaTokenizer.from_pretrained("models/codeBERT_smartContractAnalysis")
_, _, scaler = load_and_preprocess_fnn_data("data/github_data.csv")  # For FNN scaling

# Load FNN
fnn_input_dim = 4  # Based on your features
fnn = ScamDetectionFNN.load(fnn_input_dim, "models/fnn_scam_detection")

device = torch.device("cpu")  # Docker default
codebert.to(device)
fnn.to(device)
codebert.eval()
fnn.eval()

def predict_codebert(code):
    encodings = tokenizer(code, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    with torch.no_grad():
        outputs = codebert(input_ids, attention_mask)
        prediction = torch.argmax(outputs, dim=-1).item()
    return "good" if prediction == 0 else "bad"

def predict_fnn(features):
    features_tensor = torch.FloatTensor(scaler.transform([features])).to(device)
    with torch.no_grad():
        output = fnn(features_tensor)
        prediction = "scam" if output.item() > 0.5 else "not scam" #Adjust if need range as output for scam meter
    return prediction

@app.route("/predict_code", methods=["POST"])
def predict_code_endpoint():
    data = request.get_json()
    code = data.get("code")
    if not code:
        return jsonify({"error": "No code provided"}), 400
    prediction = predict_codebert(code)
    return jsonify({"prediction": prediction})

@app.route("/predict_scam", methods=["POST"])
def predict_scam_endpoint():
    data = request.get_json()
    features = data.get("features")  # Expect [Chain, Losses, Type, Root Causes]
    if not features or len(features) != 4:
        return jsonify({"error": "Invalid features"}), 400
    prediction = predict_fnn(features)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)