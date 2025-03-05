#Added by Palladen, for using tuned roBERTa model

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from scipy.special import softmax

# Load the fine-tuned model and tokenizer
model_path = 'roBERTa_fine_tuned.pth'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)  # Adjust num_labels if necessary
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def predict_sentiment(text):
    # Tokenize the input text
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move the encoded text to the same device as the model
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted class probabilities
    probs = softmax(outputs.logits.cpu().numpy()[0])

    # Map probabilities to sentiment labels
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    sentiment = sentiment_labels[probs.argmax()]
    confidence = probs.max()

    return sentiment, confidence


# Example usage
texts = [
    "I absolutely loved this movie!",
    "The service at the restaurant was terrible.",
    "The weather today is just okay.",
]

for text in texts:
    sentiment, confidence = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}")
    print()
