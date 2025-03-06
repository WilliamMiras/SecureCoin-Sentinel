# Combines CodeBERT and FNN outputs (assumes some feature alignment).

import torch
import torch.nn as nn
from .codebert import CodeBERTClassifier
from .fnn import ScamDetectionFNN

class HybridModel(nn.Module):
    def __init__(self, fnn_input_dim):
        super(HybridModel, self).__init__()
        self.codebert = CodeBERTClassifier(num_labels=2)
        self.fnn = ScamDetectionFNN(fnn_input_dim)
        self.classifier = nn.Linear(768 + 1, 2)  # CodeBERT (768) + FNN (1)

    def forward(self, input_ids, attention_mask, fnn_input):
        codebert_logits = self.codebert(input_ids, attention_mask)
        fnn_output = self.fnn(fnn_input)
        combined = torch.cat((codebert_logits, fnn_output), dim=1)
        logits = self.classifier(combined)
        return logits

    def save(self, path):
        self.codebert.save(path + "/codebert")
        self.fnn.save(path + "/fnn")
        torch.save(self.classifier.state_dict(), f"{path}/classifier.pt")

    @classmethod
    def load(cls, fnn_input_dim, path):
        model = cls(fnn_input_dim)
        model.codebert = CodeBERTClassifier.load(path + "/codebert")
        model.fnn = ScamDetectionFNN.load(fnn_input_dim, path + "/fnn")
        model.classifier.load_state_dict(torch.load(f"{path}/classifier.pt"))
        return model