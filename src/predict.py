import torch
import numpy as np
from src.preprocess import clean_text

# predict a single string, use the pre-trained model for classification
def predict_single_text(text, model, vectorizer, class_to_idx):
    #cleaning the text
    cleaned_text = clean_text(text)

    #vectorization (clean text -> TF-IDF vector)
    #transform, not fit-transform to use the exisitng vocabulary
    text_vectorized = vectorizer.transform([cleaned_text])

    #conversion to tensor
    input_tensor = torch.tensor(text_vectorized.toarray()).float()

    #evaluation mode - dropout off
    model.eval()
    with torch.no_grad(): #disables gradient (+ speed and memory)
        output = model(input_tensor)

    predicted_id = output.argmax(dim=1).item()

    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    predicted_class = idx_to_class[predicted_id]

    #final predicted class
    return predicted_class

