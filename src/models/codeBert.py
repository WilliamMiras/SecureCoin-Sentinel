# Defines the CodeBERT model for use standalone or within the hybrid model. Wraps the pre-trained CodeBERT with a classification head.
# Creator: Will

import torch
import torch.nn as nn
from transformers import RobertaModel

class CodeBERTClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(CodeBERTClassifier, self).__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)  # 768 is CodeBERT's hidden size

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def save(self, path):
        self.codebert.save_pretrained(path)
        torch.save(self.state_dict(), f"{path}/classifier_state.pt")

    @classmethod
    def load(cls, path):
        model = cls()
        model.codebert = RobertaModel.from_pretrained(path)
        model.load_state_dict(torch.load(f"{path}/classifier_state.pt"))
        return model