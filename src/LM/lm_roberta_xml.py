from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.plotting import plot_confusion_matrix, plot_metrics_bar_chart
from src.plotting import plot_precision_recall_curve, plot_roc_curve
import matplotlib.pyplot as plt


"""  Model initialization """

device = "cuda" if torch.cuda.is_available() else "cpu" # GPU, if available, otherwise CPU

model_name = "xlm-roberta-base"
NUM_CLASSES = 4

tokenizer = AutoTokenizer.from_pretrained(model_name)

# loading pretrained xlm-Roberta
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES,  ignore_mismatched_sizes=True )
model.to(device)


for name, param in model.named_parameters():
    if name.startswith("bert.embeddings") or name.startswith("bert.encoder.layer.0"): # freeze layer (will not be trained, prevention of overfitting)
        param.requires_grad = False 


""" Dataset class  """

class TextDataset(Dataset):
    # PyTorch dataset
    # transformation of text and labels into PyTorch tensors, tokenization, padding, truncation
    def __init__(self, texts, labels, tokenizer, max_length = 300):
       self.texts = texts 
       self.labels = labels
       self.tokenizer = tokenizer   
       self.max_length = max_length    

    def __len__(self):
        return len(self.texts)  
    
    def __getitem__(self, index):  
       text = self.texts[index]
       label = self.labels[index]
       encoding = self.tokenizer(
           text,
           truncation = True,
           padding = 'max_length',
           max_length = self.max_length,
           return_tensors = 'pt'
       )
       encoding = {k: v.squeeze(0) for k, v in encoding.items()}   
       encoding['labels']= torch.tensor(label, dtype=torch.long)    
       return encoding
    

""" Dataloader """
# conversion of text and labels into batches, shuffling of data for training, faster processing
def create_dataloaders(X_train, y_train, X_test, y_test, batch_size = 16):

    # training and test dataset
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)

    # dataloadres - creation of batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    return train_loader, test_loader


""" Training """
def train_model(model, train_loader, epochs = 3, lr = 2e-5):

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    # training mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k : v.to(device) for k, v in batch.items()}    
            outputs = model(**batch) 
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    return model


""" Evaluation """
# classification report
def evaluate_model(model, test_loader, label_names = None):
    model.eval()
    preds, labels, probs = [], [], []

    with torch.no_grad(): # 
        for batch in test_loader:
            input_batch = {k : v.to(device) for k , v in batch.items() if k != 'labels'} 
            outputs = model(**input_batch) 
            preds_batch = outputs.logits.argmax(dim=1).cpu().numpy()    
            probs_batch = F.softmax(outputs.logits, dim=1).cpu().numpy()  
            preds.extend(preds_batch)
            labels.extend(batch['labels'].numpy())
            probs.extend(probs_batch)

    from sklearn.metrics import classification_report
    print(classification_report(labels, preds, target_names=label_names))

    

    true_ids = np.asarray(labels, dtype=int)
    pred_ids = np.asarray(preds, dtype=int)

    plot_confusion_matrix(true_ids, pred_ids, label_names, prefix='roberta_')
    plot_metrics_bar_chart(true_ids, pred_ids, label_names, prefix='roberta_')





""" Single text prediction """

def predict_sentence(text, model, tokenizer, target_names, device = 'cuda', max_length = 128):
    model.to(device)
    model.eval() 

# tokenizacia textu
    encoding = tokenizer(
        text,
        truncation = True,
        padding = 'max_length',
        max_length = max_length,
        return_tensors = 'pt'
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    pred_idx = logits.argmax(dim=1).item()
    pred_label = target_names[pred_idx]

    return pred_label