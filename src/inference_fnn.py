import torch
from models.fnn import ScamDetectionFNN
from dataset import load_and_preprocess_fnn_data

# Load model
_, test_dataset, scaler = load_and_preprocess_fnn_data("../data/github_data.csv")
input_dim = test_dataset.tensors[0].shape[1]
model = ScamDetectionFNN.load(input_dim, "../models/fnn_scam_detection")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Example inference
with torch.no_grad():
    X_test, _ = test_dataset[:1]  # Take one sample
    X_test = X_test.to(device)
    output = model(X_test)
    prediction = "scam" if output.item() > 0.5 else "not scam"
    print(f"Prediction: {prediction}")