import torch
import torch.nn as nn       #neural network module
import torch.optim as optim     #optimization module
import matplotlib.pyplot as plt     #plotting library
import numpy as np  # for matrix handling

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.plotting import plot_confusion_matrix, plot_learning_curve, plot_metrics_bar_chart

import pandas as pd

#feed forward neural network
#inupt(dim) - size of the input features (5000)
#hidden(dim) - number of neurons in the hidden layer
#dropout - probability of a neuron being set to zero (only during training)

""" Simple Feed Forward NN with one hidden layer"""

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
    

""" Deep Feed Forward NN """

class DeepTextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate = 0.5):

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
    


""" Recurent NN """    
# useful just when input is embedding / tokens  

class RNNTextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 1, dropout_rate = 0.5, rnn_type=type):

        super(RNNTextClassifier, self).__init__()

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True, dropout = dropout_rate)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first = True, dropout = dropout_rate)
        else:
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first = True, nonlinearity= 'tanh')
    
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
       output, _ = self.rnn(x)
       last_output = output[:, -1, :]
       return self.fc(last_output)

        
        
""" Training the model """ 

def train_model(X_train, y_train, X_test, y_test, model_class =TextClassifier, model_params = None, epochs=5, lr=0.01):
    if model_params is None:
        model_params = {}

  

    #sparse matrices (convert to pytorch float)
    X_train = torch.tensor(X_train.toarray()).float()
    X_test = torch.tensor(X_test.toarray()).float()

    # cotegories -> integers
    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)} #category + integer id

    #training and test category labels (panda series) - map to correct ids
    y_train = torch.tensor(y_train.map(class_to_idx).values).long()
    y_test = torch.tensor(y_test.map(class_to_idx).values).long()

    #print sample of inputs
    print_NN_input_sample(X_train, y_train)

    #initialization of text classifier - used model
    output_dim = len(class_to_idx)

    model = model_class(X_train.shape[1], output_dim=output_dim, **model_params)
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
        #if(epoch % 10 == 0):
        #    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

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

    """     print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50) """

    # classification report
    report = classification_report(
        true_ids,
        preds_ids,
        target_names=target_names,
        digits = 4
    )
    #accuracy report
    print(report)


    acc = (preds == y_test).float().mean().item()
    print(f'Test accuracy: {acc:.4f}')

    """ plotting """

    #plot confusion matrix
    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='nn_')

    #plot learning curve
    plot_learning_curve(epochs, train_loss_history,test_acc_history, prefix='nn_')

    #plot metrics bar chart
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='nn_')

    return model, class_to_idx, acc


def print_NN_input_sample(X_train, y_train):
    print(f"X_train Shape (samples, features): {X_train.shape}")
    print(f"X_train Data Type: {X_train.dtype}")
    print(f"y_train Shape (labels): {y_train.shape}")
    print(f"y_train Data Type: {y_train.dtype}")
    print(f"Feature Vector Length (input_dim): {X_train.shape[1]}")
    print("-" * 30)

    print("First sample (first 10 features):")
    # Using .cpu().numpy() to ensure compatibility if a GPU is used, and for readability
    print(X_train[0][:10].cpu().numpy())
    print("-" * 30)