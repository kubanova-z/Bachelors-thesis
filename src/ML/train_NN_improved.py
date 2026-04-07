#imports
import numpy as np  # for matrix handling
import pandas as pd

import torch
import torch.nn as nn       #neural network module
import torch.optim as optim     #optimization module
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt     #plotting library
from src.plotting import plot_confusion_matrix, plot_learning_curve, plot_metrics_bar_chart




#models
""" Simple Feed Forward NN with one hidden layer"""

class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate = 0.2):
        super(TextClassifier, self).__init__()
        # 1. layer - Linear
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 2. activation function - relu
        self.relu = nn.ReLU()

        # 3. dropout layer (prevent overfitting)
        self.dropout = nn.Dropout(p=dropout_rate)

        # 4. output layer (raw scores for each category)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

#operations on input tensor (data structure)
    def forward(self, x):
        # 1. layer
        x = self.fc1(x)
        # 2. relu activation function
        x = self.relu(x)
        # 3. dropout layer
        x = self.dropout(x)
        # 4. output layer
        return self.fc2(x)
    

""" Deep Feed Forward NN """

class DeepTextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate = 0.2):

        super(DeepTextClassifier, self).__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    


""" Focal Loss 
- model sa viac sustredi na "tazke" vzorky z minortinej triedy 
- alpha = vahy pre triedy 
- gamma = ako velmi sa sustredit na tazke vzorky a ignorovat jednoduche (gamma = 0 -> klasicky cross entropy, gamma > 0 -> viac fokus na tazke vzorky)
"""    


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = alpha
            self.alpha_is_scalar = True
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))  # <-- register_buffer keeps it with the model
            self.alpha_is_scalar = False
        self.gamma = gamma
        self.reduction = reduction
       

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none') # cross entropy loss (vrati CE pre vzorku - batch)
        pt = torch.exp(-ce_loss)    # konverzia CE na pravdepodobnost (s akou pravdepodobnostou je modla modelu trieda spravna)
        if self.alpha_is_scalar:    # aplha = scalar, rovnake vahy pre vsetky triedy
            alpha_factor = self.alpha
        else:
            # alpha = tensor - balansovanie tried 
            # triede 1 sme nastavili vyssiu vahu ako 0 triede
            alpha_factor = self.alpha.to(targets.device)[targets]  

            # gamma - fokus na tazsie vzorky

        focal_loss = alpha_factor * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

def to_tensor(X):
    """Convert TF-IDF sparse matrix or numpy array to torch tensor"""
    if hasattr(X, "toarray"):
        return torch.tensor(X.toarray(), dtype=torch.float32)
    return torch.tensor(X, dtype=torch.float32)

def encode_labels(y, class_to_idx=None):
    if class_to_idx is None:
        classes = sorted(list(set(y)))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_encoded = torch.tensor(y.map(class_to_idx).values).long()
    return y_encoded, class_to_idx


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32, sampler=None):

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler # weighted sampler
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


#training

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_loss = 0

    for xb, yb in loader:

        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()

        outputs = model(xb)

        loss = criterion(outputs, yb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0
    preds = []
    labels = []

    with torch.no_grad():

        for xb, yb in loader:

            xb, yb = xb.to(device), yb.to(device)

            outputs = model(xb)

            p = outputs.argmax(dim=1)

            correct += (p == yb).sum().item()
            total += yb.size(0)

            preds.extend(p.cpu().numpy())
            labels.extend(yb.cpu().numpy())

    acc = correct / total

    return acc, preds, labels



""" 
Class weights
- aplikovane priamo do chybovej funkcie
- ked model nespravne klasifikuje vzorku z minoritnej triedy, strata bude vacsia ako pri nespravnej klasifikacii majoritnej triedy
- model je nuteny venovat viac pozornosti minoritnej triede, aby minimalizoval celkovu stratu
- nemenia sa vstupne data do modelu, ale model sa uci zohladnovat nerovnovahu v chybovej funkcii
"""

def compute_weights(y_train, boost_factor=1.0, device="cpu", soften = 0.5):

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train.cpu().numpy()),
        y=y_train.cpu().numpy()
    )

    class_weights = 1 + (class_weights - 1) * soften
    if len(class_weights) > 1:
        class_weights[1] *= boost_factor

    return torch.tensor(class_weights, dtype=torch.float32).to(device)


""" 
Weighted Sampler
- sampler, ktory zabezpeci, ze sa vzorky z minoritnej triedy budu mmodelu zobrazovat castejsie
- aplikuje sa do dataloaderu, ktory sa pouziva pri trenovani modelu (pred vypoctom chyby)
- v kazdej epoche model vidi viac vzoriek z minoritnej triedy


"""

def create_sampler(y_train):

    class_sample_counts = np.bincount(y_train.cpu().numpy())
    print(f"Class counts: {class_sample_counts}")  

    weights = 1. / class_sample_counts
    print(f"Sample weights per class: {weights}")

    sample_weights = weights[y_train.cpu().numpy()]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    model_class=TextClassifier,
    model_params=None,
    epochs=10,
    lr=1e-3,
    batch_size=32,
    class_weights=False,
    sampler=None,
    loss_fn=None,
    use_scheduler=False
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = to_tensor(X_train)
    X_test = to_tensor(X_test)

    y_train, class_to_idx = encode_labels(y_train)
    y_test, _ = encode_labels(y_test, class_to_idx)

    if sampler == "weighted":
        sampler = create_sampler(y_train)

    train_loader, test_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size, sampler
    )

    model = model_class(
        input_dim=X_train.shape[1],
        output_dim=len(class_to_idx),
        **(model_params or {})
    ).to(device)

    if loss_fn is None:
        # class weights
        if class_weights:
            weights = compute_weights(y_train, device=device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

    else:
        criterion = loss_fn

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        if use_scheduler else None
    )

    train_losses = []
    test_accs = []

    for epoch in range(epochs):

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        acc, preds, labels = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        test_accs.append(acc)

        if scheduler is not None:
            scheduler.step(acc)

            # optional: print LR changes so you can see it working
            current_lr = optimizer.param_groups[0]['lr']
            #print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | LR: {current_lr:.6f}")


    


    return model, class_to_idx, preds, labels, train_losses, test_accs


