# %%
import torch
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


from src.LM.lm_slovakBERT import (
    tokenizer, model,
    create_dataloaders,
    train_model,
    evaluate_model,
    get_class_weights
)

from src.data_loader import load_data
from src.preprocess import clean_text


# %%
df = load_data("/home/xkubanova_126831/bakalarka/nbs_binary/data/ekosentiment_titles_binary.csv")
df['Text'] = df['Text'].apply(clean_text)  

le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

# ulozenie katagorii
target_names = le.classes_.tolist()

print("Target names:", target_names)
print("Label distribution:")
print(df["Category_encoded"].value_counts())





# %%
# rozdelenie datasetu
X_train_text, X_test_text, y_train_encoded, y_test_encoded = train_test_split(
    df['Text'].tolist(), 
    df['Category_encoded'].tolist(),
    test_size=0.2, 
    random_state=42,
    stratify=df['Category_encoded']
)
print("Train size:", len(X_train_text))
print("Test size:", len(X_test_text))

# %%
train_loader, test_loader = create_dataloaders(
    X_train_text,
    y_train_encoded,
    X_test_text,
    y_test_encoded,
    batch_size=16
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_weights = get_class_weights(y_train_encoded, device)
print("Class weights:", class_weights)


# %%
trained_model = train_model(
    model,
    train_loader,
    epochs=5,
    lr=1e-5, # nizsi LR !
    use_focal_loss=True,
    class_weights=class_weights
)

trained_model.save_pretrained("models/slovakBert_binary")
tokenizer.save_pretrained("models/slovakBert_binary")

evaluate_model(
    trained_model,
    test_loader,
    label_names=target_names
)



