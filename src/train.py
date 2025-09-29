import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
import pandas as pd

class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate = 0.5):
        super(TextClassifier, self).__init__()
        # 1. layer - Linear
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 2. activation function - relu
        self.relu = nn.ReLU()

        # 3. dropout layer (prevent overfitting)
        self.dropout = nn.Dropout(p=dropout_rate)

        # 4. output layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # 1. layer
        x = self.fc1(x)
        # 2. relu activation function
        x = self.relu(x)
        # 3. dropout layer
        x = self.dropout(x)
        # 4. output layer
        return self.fc2(x)
        
        

def train_model(X_train, y_train, X_test, y_test, epochs=5, lr=0.01):
    X_train = torch.tensor(X_train.toarray()).float()
    X_test = torch.tensor(X_test.toarray()).float()

    # cotegories -> integers
    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    y_train = torch.tensor(y_train.map(class_to_idx).values).long()
    y_test = torch.tensor(y_test.map(class_to_idx).values).long()

    model = TextClassifier(X_train.shape[1], 128, len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        with torch.no_grad():
            preds = model(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()
            print(f'Test accuracy: {acc:.4f}')
            
    with torch.no_grad():
        # raw predictions
        outputs = model(X_test)

        # predicted classes ids
        preds_ids = outputs.argmax(dim=1).cpu().numpy()

        # true class ids
        true_ids = y_test.cpu().numpy()

# inverse mapping for labels

    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)

    # classification report
    report = classification_report(
        true_ids,
        preds_ids,
        target_names=target_names,
        digits = 4
    )
    print(report)

    return model, class_to_idx