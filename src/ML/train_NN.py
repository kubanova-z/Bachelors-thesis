from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn       #neural network module
import torch.optim as optim     #optimization module
import matplotlib.pyplot as plt     #plotting library
import numpy as np  # for matrix handling
from torch.utils.data import WeightedRandomSampler

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.plotting import plot_confusion_matrix, plot_learning_curve, plot_metrics_bar_chart

import pandas as pd

#feed forward neural network
#inupt(dim) - size of the input features (5000)
#hidden(dim) - number of neurons in the hidden layer
#dropout - probability of a neuron being set to zero (only during training)

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

        
        
""" Training the model with class weights""" 

def train_model(X_train, y_train, X_test, y_test, model_class =TextClassifier, model_params = None, epochs=5, lr=0.01):
    if model_params is None:
        model_params = {}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # convert inputs to PyTorch tensors - tp be able to handle both TF-IDF and embeddings

    if hasattr(X_train, "toarray"): # TF-IDF
        X_train_tensor = torch.tensor(X_train.toarray()).float()
    else:   #embedding
        X_train_tensor = torch.tensor(X_train).float()
        

    if hasattr(X_test, "toarray"): # TF-IDF
        X_test_tensor = torch.tensor(X_test.toarray()).float()
    else:   #embedding
        X_test_tensor = torch.tensor(X_test).float()


    #sparse matrices (convert to pytorch float)
    if hasattr(X_train, "toarray"):  # e.g. TF-IDF sparse matrix
        X_train = torch.tensor(X_train.toarray()).float()
    else:  # e.g. Word2Vec numpy array
        X_train = torch.tensor(X_train).float()

    if hasattr(X_test, "toarray"):
        X_test = torch.tensor(X_test.toarray()).float()
    else:
        X_test = torch.tensor(X_test).float()

    # cotegories -> integers
    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)} #category + integer id

    # class_weights = compute_class_weight(
    #     class_weight='balanced',
    #     classes=np.array(classes),
    #     y=y_train.values   # convert tensor to numpy array
    # )

    # boost_factor = 1.3
    # class_weights[1] *= boost_factor
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # print("Class weights:", class_weights)


    #training and test category labels (panda series) - map to correct ids
    y_train = torch.tensor(y_train.map(class_to_idx).values).long()
    y_test = torch.tensor(y_test.map(class_to_idx).values).long()

    # class weights

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(classes),
        y=y_train.numpy()   # convert tensor to numpy array
    )

    boost_factor = 2
    class_weights[1] *= boost_factor
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights)




    # class_sample_counts = np.bincount(y_train.numpy())
    # weights = 1. / class_sample_counts
    # sample_weights = weights[y_train.numpy()]

    # sampler = WeightedRandomSampler(
    #     weights=sample_weights,
    #     num_samples=len(sample_weights),
    #     replacement=True
    # )


    batch_size = 16

    #dataloaders for batched training
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    
    # train_loader = DataLoader(
    # train_ds,
    # batch_size=batch_size,
    # sampler=sampler,   # balance the batches
    # )

    

   
    #print sample of inputs
    #print_NN_input_sample(X_train, y_train)

    #initialization of text classifier - used model
    input_dim = X_train_tensor.shape[1]
    output_dim = len(class_to_idx)
    

   
    model = model_class(input_dim, output_dim=output_dim, **model_params)
    model = model.to(device)

    #loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #accuracy and loss

    train_loss_history = []
    test_acc_history = []

    


    #TRAINING
    for epoch in range(epochs):
        model.train()   #set to training mode (enable dropout)
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()   #reset gradients
            outputs = model(xb)    #predicted outputs
            loss = criterion(outputs, yb) #calculate loss
            loss.backward() #backward pass - algoritmus spatneho sirenia chyby
            optimizer.step()    #update model weights based on gradients
            
            epoch_loss += loss.item() * xb.size(0) 

        avg_epoch_loss = epoch_loss / len(train_loader.dataset)  # average loss per sample
        train_loss_history.append(avg_epoch_loss)
        #if(epoch % 10 == 0):
        #    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    
        model.eval()    #set to evaluation mode (disable dropout)
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = outputs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        test_acc_history.append(acc)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds_ids = outputs.argmax(dim=1).cpu().numpy()
        true_ids = y_test.cpu().numpy()

    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    target_names = [str(idx_to_class[i]) for i in sorted(idx_to_class.keys())]

    print("\nClassification Report:")
    print(classification_report(true_ids, preds_ids, target_names=target_names, digits=4))

    final_acc = (preds_ids == true_ids).mean()
    print(f"Final Test Accuracy: {final_acc:.4f}")

        
#     print(f'Test accuracy: {acc:.4f}')



#     #final accuracy check after all epochs
#     with torch.no_grad():
#         model.eval()

#         X_test = X_test.to(device)
#         y_test = y_test.to(device)
#         # raw predictions
#         outputs = model(X_test)

#         # predicted classes ids
#         preds_ids = outputs.argmax(dim=1).cpu().numpy()

#         # true class ids
#         true_ids = y_test.cpu().numpy()

# # inverse mapping for labels / for readable report

#     idx_to_class = {i: cls for cls, i in class_to_idx.items()}
#     target_names = [str(idx_to_class[i]) for i in sorted(idx_to_class.keys())]


#     """     print("\n" + "="*50)
#     print("CLASSIFICATION REPORT")
#     print("="*50) """

#     # classification report
#     report = classification_report(
#         true_ids,
#         preds_ids,
#         target_names=target_names,
#         digits = 4
#     )
#     #accuracy report
#     print(report)


#     final_preds = outputs.argmax(dim=1)
#     acc = (final_preds == y_test).float().mean().item()

#     print(f'Test accuracy: {acc:.4f}')

    """ plotting """

    #plot confusion matrix
    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='nn_')

    #plot learning curve
    plot_learning_curve(epochs, train_loss_history,test_acc_history, prefix='nn_')

    #plot metrics bar chart
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='nn_')

    return model, class_to_idx, acc


""" Training the model with sample weights """ 

def train_model_sample_weights(X_train, y_train, X_test, y_test, model_class =TextClassifier, model_params = None, epochs=5, lr=0.01):
    if model_params is None:
        model_params = {}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # convert inputs to PyTorch tensors - tp be able to handle both TF-IDF and embeddings

    if hasattr(X_train, "toarray"): # TF-IDF
        X_train_tensor = torch.tensor(X_train.toarray()).float()
    else:   #embedding
        X_train_tensor = torch.tensor(X_train).float()
        

    if hasattr(X_test, "toarray"): # TF-IDF
        X_test_tensor = torch.tensor(X_test.toarray()).float()
    else:   #embedding
        X_test_tensor = torch.tensor(X_test).float()


    #sparse matrices (convert to pytorch float)
    if hasattr(X_train, "toarray"):  # e.g. TF-IDF sparse matrix
        X_train = torch.tensor(X_train.toarray()).float()
    else:  # e.g. Word2Vec numpy array
        X_train = torch.tensor(X_train).float()

    if hasattr(X_test, "toarray"):
        X_test = torch.tensor(X_test.toarray()).float()
    else:
        X_test = torch.tensor(X_test).float()

    # cotegories -> integers
    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)} #category + integer id


    #training and test category labels (panda series) - map to correct ids
    y_train = torch.tensor(y_train.map(class_to_idx).values).long()
    y_test = torch.tensor(y_test.map(class_to_idx).values).long()

    # class weights fo samples
    class_sample_counts = np.bincount(y_train.numpy())
    weights = 1. / class_sample_counts
    sample_weights = weights[y_train.numpy()]


    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


    batch_size = 16

    #dataloaders for batched training
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)


    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler = sampler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
     

   
    #print sample of inputs
    #print_NN_input_sample(X_train, y_train)

    #initialization of text classifier - used model
    input_dim = X_train_tensor.shape[1]
    output_dim = len(class_to_idx)
    

   
    model = model_class(input_dim, output_dim=output_dim, **model_params)
    model = model.to(device)

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
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()   #reset gradients
            outputs = model(xb)    #predicted outputs
            loss = criterion(outputs, yb) #calculate loss
            loss.backward() #backward pass - algoritmus spatneho sirenia chyby
            optimizer.step()    #update model weights based on gradients
            
            epoch_loss += loss.item() * xb.size(0)
        train_loss_history.append(epoch_loss / len(train_loader.dataset)) 

       

    
        model.eval()    #set to evaluation mode (disable dropout)
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = outputs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        test_acc_history.append(acc)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds_ids = outputs.argmax(dim=1).cpu().numpy()
        true_ids = y_test.cpu().numpy()

    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    target_names = [str(idx_to_class[i]) for i in sorted(idx_to_class.keys())]

    print("\nClassification Report:")
    print(classification_report(true_ids, preds_ids, target_names=target_names, digits=4))

    final_acc = (preds_ids == true_ids).mean()
    print(f"Final Test Accuracy: {final_acc:.4f}")

     
    """ plotting """

    #plot confusion matrix
    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='nn_')

    #plot learning curve
    plot_learning_curve(epochs, train_loss_history,test_acc_history, prefix='nn_')

    #plot metrics bar chart
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='nn_')

    return model, class_to_idx, acc




def train_model_class_weights_and_sampling(X_train, y_train, X_test, y_test, model_class =TextClassifier, model_params = None, epochs=5, lr=0.01):
    if model_params is None:
        model_params = {}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # convert inputs to PyTorch tensors - tp be able to handle both TF-IDF and embeddings

    if hasattr(X_train, "toarray"): # TF-IDF
        X_train_tensor = torch.tensor(X_train.toarray()).float()
    else:   #embedding
        X_train_tensor = torch.tensor(X_train).float()
        

    if hasattr(X_test, "toarray"): # TF-IDF
        X_test_tensor = torch.tensor(X_test.toarray()).float()
    else:   #embedding
        X_test_tensor = torch.tensor(X_test).float()


    #sparse matrices (convert to pytorch float)
    if hasattr(X_train, "toarray"):  # e.g. TF-IDF sparse matrix
        X_train = torch.tensor(X_train.toarray()).float()
    else:  # e.g. Word2Vec numpy array
        X_train = torch.tensor(X_train).float()

    if hasattr(X_test, "toarray"):
        X_test = torch.tensor(X_test.toarray()).float()
    else:
        X_test = torch.tensor(X_test).float()

    # cotegories -> integers
    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)} #category + integer id

    # class_weights = compute_class_weight(
    #     class_weight='balanced',
    #     classes=np.array(classes),
    #     y=y_train.values   # convert tensor to numpy array
    # )

    # boost_factor = 1.3
    # class_weights[1] *= boost_factor
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # print("Class weights:", class_weights)


    #training and test category labels (panda series) - map to correct ids
    y_train = torch.tensor(y_train.map(class_to_idx).values).long()
    y_test = torch.tensor(y_test.map(class_to_idx).values).long()

    # class weights

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(classes),
        y=y_train.numpy()   # convert tensor to numpy array
    )

    boost_factor = 2
    class_weights[1] *= boost_factor
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights)


    batch_size = 16

    #dataloaders for batched training
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    class_sample_counts = np.bincount(y_train.numpy())
    weights = 1. / class_sample_counts
    sample_weights = weights[y_train.numpy()]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )



    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler = sampler)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    
   
    #print sample of inputs
    #print_NN_input_sample(X_train, y_train)

    #initialization of text classifier - used model
    input_dim = X_train_tensor.shape[1]
    output_dim = len(class_to_idx)
    

   
    model = model_class(input_dim, output_dim=output_dim, **model_params)
    model = model.to(device)

    #loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #accuracy and loss

    train_loss_history = []
    test_acc_history = []

    


    #TRAINING
    for epoch in range(epochs):
        model.train()   #set to training mode (enable dropout)
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()   #reset gradients
            outputs = model(xb)    #predicted outputs
            loss = criterion(outputs, yb) #calculate loss
            loss.backward() #backward pass - algoritmus spatneho sirenia chyby
            optimizer.step()    #update model weights based on gradients
            
            epoch_loss += loss.item() * xb.size(0) 

        avg_epoch_loss = epoch_loss / len(train_loader.dataset)  # average loss per sample
        train_loss_history.append(avg_epoch_loss)
       

    
        model.eval()    #set to evaluation mode (disable dropout)
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = outputs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        test_acc_history.append(acc)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds_ids = outputs.argmax(dim=1).cpu().numpy()
        true_ids = y_test.cpu().numpy()

    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    target_names = [str(idx_to_class[i]) for i in sorted(idx_to_class.keys())]

    print("\nClassification Report:")
    print(classification_report(true_ids, preds_ids, target_names=target_names, digits=4))

    final_acc = (preds_ids == true_ids).mean()
    print(f"Final Test Accuracy: {final_acc:.4f}")


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