import torch
import torch.nn as nn       #neural network module
import torch.optim as optim     #optimization module
import matplotlib.pyplot as plt     #plotting library
import numpy as np  # for matrix handling

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd

#feed forward neural network
#inupt(dim) - size of the input features (5000)
#hidden(dim) - number of neurons in the hidden layer
#dropout - probability of a neuron being set to zero (only during training)

class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate = 0.5):
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
        
        
#set up the model, train on vectorized data and evaluate

def train_model(X_train, y_train, X_test, y_test, epochs=5, lr=0.01):
    #sparse matrices (convert to pytorch float)
    X_train = torch.tensor(X_train.toarray()).float()
    X_test = torch.tensor(X_test.toarray()).float()

    # cotegories -> integers
    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)} #category + integer id

    #training and test category labels (panda series) - map to correct ids
    y_train = torch.tensor(y_train.map(class_to_idx).values).long()
    y_test = torch.tensor(y_test.map(class_to_idx).values).long()

    #initialization of text classifier
    model = TextClassifier(X_train.shape[1], 128, len(classes))
    #loss function
    criterion = nn.CrossEntropyLoss()
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #accuracy and loss

    train_loss_history = []
    test_acc_history = []


    #TRAINING
    for epoch in range(epochs):
        model.train()   #set to training mode (enable dropout)
        optimizer.zero_grad()   #reset gradients
        outputs = model(X_train)    #predicted outputs
        loss = criterion(outputs, y_train) #calculate loss
        loss.backward() #backward pass - algoritmus spatneho sirenia chyby
        optimizer.step()    #update model weights based on gradients
        train_loss_history.append(loss.item())  #trauning loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        #test accuracy check
        with torch.no_grad():
            model.eval()    #set to evaluation mode (disable dropout)
            preds = model(X_test).argmax(dim=1) #predicted ids
            acc = (preds == y_test).float().mean().item()   #accuracy (correct / total samples)
            test_acc_history.append(acc)
            print(f'Test accuracy: {acc:.4f}')

    #final accuracy check after all epochs
    with torch.no_grad():
        model.eval()
        # raw predictions
        outputs = model(X_test)

        # predicted classes ids
        preds_ids = outputs.argmax(dim=1).cpu().numpy()

        # true class ids
        true_ids = y_test.cpu().numpy()

# inverse mapping for labels / for readable report

    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())] #target names sorted according to ids

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
    #accuracy report
    print(report)

    #plot confusion matrix
    plot_confusion_matrix(true_ids, preds_ids, target_names)

    #plot learning curve
    plot_learning_curve(epochs, train_loss_history,test_acc_history)

    #plot metrics bar chart
    plot_metrics_bar_chart(true_ids, preds_ids, target_names)

    return model, class_to_idx


#CONFUSION MATRIX
def plot_confusion_matrix(true_ids, preds_ids, target_names):
    cm = confusion_matrix(true_ids, preds_ids)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #display object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=target_names)

    #plot matrix
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(
        cmap=plt.cm.PuBuGn, 
        ax=ax, 
        xticks_rotation='vertical',
        values_format='.1%')

    ax.set_title('Confusion matrix', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Category', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
    
    plt.tight_layout()

    #plt.show()
    plt.savefig('confusion_matrix_results.png') 
    plt.close(fig) # Close the figure to free up memory
    
    print("\n[INFO] Confusion Matrix saved as confusion_matrix_results.png in the current directory.")


#LEARNING CURVE
def plot_learning_curve(epochs, train_loss, test_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    epoch_range = range(1, epochs+1)

    ax1.plot(epoch_range, train_loss, label='Training loss', color='red')
    ax1.set_title('Training loss per epoch', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Cross Entropy)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)


    ax2.plot(epoch_range, test_acc, label='Test accuracy', color='blue')
    ax2.set_title('Test accuracy per Epoch', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.close(fig)

    
# bar chart - F1 score
def plot_metrics_bar_chart(true_ids, preds_ids, target_names):

    #classification report as dictionary
    report = classification_report(
        true_ids, preds_ids, target_names=target_names, output_dict=True
    )

    # report dictionary -> DataFrame 
    # transpose, exclude final rows
    df_report = pd.DataFrame(report).transpose().iloc[:-3]

    fig, ax = plt.subplots(figsize=(10,6))

    df_report[['precision', 'recall', 'f1-score']].plot(
        kind='bar',
        ax=ax,
        rot=45,
        cmap=plt.cm.PuBuGn,
        edgecolor = 'black'
    )

    ax.set_title('Precision, Recall, F1', fontsize=14)
    ax.set_ylabel('Score', fontsize =12)
    ax.set_xlabel('Category', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(axis = 'y', linestyle = '--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('per_class_metrics.png', dpi=300)
    plt.close(fig)

    