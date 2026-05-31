from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from src.plotting import plot_confusion_matrix, plot_metrics_bar_chart
from src.plotting import plot_precision_recall_curve, plot_roc_curve
from torch.utils.data import WeightedRandomSampler


import matplotlib.pyplot as plt


import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""  Model initialization """

device = "cuda" if torch.cuda.is_available() else "cpu" # GPU, ked je dostupne, inak CPU

model_name = "xlm-roberta-base"
NUM_CLASSES = 2

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# load pretrained xlm-Roberta
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES)
model.to(device)




for name, param in model.named_parameters():
    if name.startswith("bert.embeddings") or name.startswith("bert.encoder.layer.0"): # freeze layer - will not be trained, prevention of overfitting, faster training
        param.requires_grad = False 


""" Dataset class  """

class TextDataset(Dataset):

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

     # class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = class_weights[y_train]
    # sampler - to handle imbalance
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


    # dataloadres - creation of batches, shuffling of data every epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    return train_loader, test_loader


''' Focal loss '''
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight = self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_class_weights(labels, device):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    return class_weights_tensor

""" Training """
def train_model(model, train_loader, epochs = 5, lr = 2e-5, use_focal_loss = True, class_weights = None):

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    model.train()

    if use_focal_loss:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=2.5,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            batch = {k : v.to(device) for k, v in batch.items()}    # use GPU
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                
            )    
            loss = criterion(outputs.logits, batch['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
       

    return model


""" Evaluation """


def evaluate_model(model, test_loader, label_names = None):
    model.eval()
    preds, labels, probs = [], [], []

    


    with torch.no_grad(): 
        for batch in test_loader:
            input_batch = {k : v.to(device) for k , v in batch.items() if k != 'labels'} 
            outputs = model(**input_batch) 

            probabilities = F.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            #predictions = outputs.logits.argmax(dim=1).cpu().numpy()   

            probs.extend(probabilities)
            labels.extend(batch['labels'].numpy())
    labels = np.array(labels)
    probs = np.array(probs)

    precision, recall, thresholds = precision_recall_curve(labels, probs)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])

    best_threshold_f1 = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Best threshold (F1-optimal): {best_threshold_f1:.3f}")
    print(f"Best F1 score: {best_f1:.3f}")

    # maximizing recall for minority class
    min_precision = 0.5 

    valid_mask = precision[:-1] >= min_precision
    if valid_mask.any():
        valid_recalls = np.where(valid_mask, recall[:-1], 0)
        best_idx_recall = np.argmax(valid_recalls)
        best_threshold_recall = thresholds[best_idx_recall]

        print(f"\nRecall-optimal threshold: {best_threshold_recall:.3f}")
        print(f"F1: {f1_scores[best_idx_recall]:.3f} | Precision: {precision[best_idx_recall]:.3f} | Recall: {recall[best_idx_recall]:.3f}")
    else:
        print(f"No valid threshold found with Precision >= {min_precision}")
        best_threshold_recall = best_threshold_f1  


    # maximizing F1 score or recall for minority class
    best_treshold = best_threshold_recall

    preds = (probs >= best_treshold).astype(int)

            # preds_batch = outputs.logits.argmax(dim=1).cpu().numpy()    
            # preds.extend(preds_batch)
            # labels.extend(batch['labels'].numpy())

    if label_names is not None:
        label_names = [str(x) for x in label_names]


    from sklearn.metrics import classification_report
    print(classification_report(labels, preds, target_names=label_names))


    

    true_ids = labels.astype(int)
    pred_ids = preds.astype(int)

    plot_confusion_matrix(true_ids, pred_ids, label_names, prefix='roberta_')
    plot_metrics_bar_chart(true_ids, pred_ids, label_names, prefix='roberta_')

    plot_roc_curve(labels, probs, prefix='roberta_')
    plot_precision_recall_curve(labels, probs, prefix='roberta_')
   

""" Single text prediction """

def predict_sentence(text, model, tokenizer, target_names, device = 'cuda', max_length = 128):
    model.to(device)
    model.eval() 

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