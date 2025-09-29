import torch
import numpy as np
from src.preprocess import clean_text

def predict_single_text(text, model, vectorizer, class_to_idx):
    cleaned_text = clean_text(text)

    text_vectorized = vectorizer.transform([cleaned_text])

    input_tensor = torch.tensor(text_vectorized.toarray()).float()

    #evaluation mode - dropout off
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    predicted_id = output.argmax(dim=1).item()

    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    predicted_class = idx_to_class[predicted_id]

    return predicted_class

