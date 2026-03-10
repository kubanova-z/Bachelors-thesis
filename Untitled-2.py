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


# %%
# %%
# === ATTENTION VISUALIZATION ===
import numpy as np
import matplotlib.pyplot as plt

def plot_avg_positional_attention(model, test_loader, device, layer=-1):
    model.eval()
    all_cls_attentions = []

    with torch.no_grad():
        for batch in test_loader:
            input_batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**input_batch, output_attentions=True)

            attn_layer = outputs.attentions[layer]           # [batch, heads, seq, seq]
            cls_attn = attn_layer[:, :, 0, :].mean(dim=1)   # [batch, seq]
            all_cls_attentions.append(cls_attn.cpu().numpy())

    all_cls_attentions = np.vstack(all_cls_attentions)
    avg_attn = all_cls_attentions.mean(axis=0)
    avg_attn = avg_attn / avg_attn.sum()

    plt.figure(figsize=(14, 4))
    plt.plot(avg_attn)
    plt.xlabel("Token Position")
    plt.ylabel("Average Attention Weight")
    plt.title(f"Average [CLS] Attention over Dataset — Layer {layer}")
    plt.tight_layout()
    plt.show()


def plot_classwise_attention(model, test_loader, device, target_names, layer=-1):
    model.eval()
    attn_by_class = {0: [], 1: []}

    with torch.no_grad():
        for batch in test_loader:
            input_batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**input_batch, output_attentions=True)

            attn_layer = outputs.attentions[layer]
            cls_attn = attn_layer[:, :, 0, :].mean(dim=1).cpu().numpy()
            preds = outputs.logits.argmax(dim=1).cpu().numpy()

            for i, pred in enumerate(preds):
                attn_by_class[pred].append(cls_attn[i])

    plt.figure(figsize=(14, 5))
    for cls_idx, cls_name in enumerate(target_names):
        attns = np.array(attn_by_class[cls_idx])
        avg = attns.mean(axis=0)
        avg = avg / avg.sum()
        plt.plot(avg, label=cls_name)

    plt.xlabel("Token Position")
    plt.ylabel("Average Attention Weight")
    plt.title(f"Average [CLS] Attention by Predicted Class — Layer {layer}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
plot_avg_positional_attention(trained_model, test_loader, device, layer=-1)
plot_classwise_attention(trained_model, test_loader, device, target_names, layer=-1)

# %%
# ================================
# CLS ATTENTION HEATMAP (WORD-LEVEL)
# ================================

def merge_subtokens(tokens, attentions):
    merged_tokens = []
    merged_attn = []

    current_word = ""
    current_attn = 0.0

    for token, attn in zip(tokens, attentions):

        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # WordPiece continuation
        if token.startswith("##"):
            current_word += token[2:]
            current_attn += attn
        else:
            if current_word != "":
                merged_tokens.append(current_word)
                merged_attn.append(current_attn)

            current_word = token
            current_attn = attn

    # Add last word
    if current_word != "":
        merged_tokens.append(current_word)
        merged_attn.append(current_attn)

    return merged_tokens, merged_attn


def plot_cls_heatmap(text, model, tokenizer, device, layer=-1, max_length=256):

    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding, output_attentions=True)

    # Get last-layer attention
    attentions = outputs.attentions[layer]  # shape: [1, heads, seq_len, seq_len]

    # CLS -> all tokens
    cls_attn = attentions[0, :, 0, :]       # [heads, seq_len]
    cls_attn = cls_attn.mean(dim=0).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    # Normalize attention
    cls_attn = cls_attn / cls_attn.sum()

    # Merge WordPiece subtokens
    words, word_attn = merge_subtokens(tokens, cls_attn)

    word_attn = np.array(word_attn)
    word_attn = word_attn / word_attn.sum()

    # Plot heatmap
    plt.figure(figsize=(max(12, len(words) * 0.4), 2.5))
    plt.imshow([word_attn], aspect="auto")
    plt.yticks([])
    plt.xticks(range(len(words)), words, rotation=90)
    plt.colorbar(label="Attention Weight")
    plt.title("CLS Attention Heatmap (Last Layer)")
    plt.tight_layout()
    plt.show()


# ================================
# Example usage
# ================================

sample_text = X_test[0]   # change index as you like
print("Sample text:", sample_text)
plot_cls_heatmap(sample_text, model, tokenizer, device)




