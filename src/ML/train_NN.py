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
    


""" Focal Loss """    
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = alpha
            self.alpha_is_scalar = True
        else:
            self.alpha = alpha
            self.alpha_is_scalar = False
        self.gamma = gamma
        self.reduction = reduction
       

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha_is_scalar:
            alpha_factor = self.alpha
        else:
            # pick alpha per sample
            alpha_factor = self.alpha[targets]  # [batch_size]

        focal_loss = alpha_factor * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
        
        
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


def train_minilm_nn(X_train, y_train, X_test, y_test, model_class=TextClassifier, model_params = None, epochs=50, lr=1e-4, batch_size=32, boost_factor=2.0):
    if model_params is None:
        model_params = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #normalize embeddings
    from sklearn.preprocessing import normalize
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    X_train = torch.tensor(X_train, dtype = torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype = torch.float32).to(device)


    #labels
    classes = sorted(list(set(y_train)))
    class_to_idx = {cls: i for i, cls in enumerate(classes)} #category + integer id

    y_train = torch.tensor(y_train.map(class_to_idx).values).long().to(device)
    y_test = torch.tensor(y_test.map(class_to_idx).values).long().to(device)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(classes),
        y=y_train.cpu().numpy()   # convert tensor to numpy array
    )
    class_weights[1] *= boost_factor
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights)

    # dataloaders
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    #model
    input_dim = X_train.shape[1]
    output_dim = len(class_to_idx)

    model = model_class(input_dim=X_train.shape[1], output_dim=len(class_to_idx), **model_params).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    test_acc_history = []

    #training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()    

            epoch_loss += loss.item() * xb.size(0)

        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        train_loss_history.append(avg_epoch_loss)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = outputs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)

        test_acc = correct / total
        test_acc_history.append(test_acc)

        
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds_ids = outputs.argmax(dim=1).cpu().numpy()

    true_ids = y_test.cpu().numpy()
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    target_names = [str(idx_to_class[i]) for i in sorted(idx_to_class.keys())]

    print(classification_report(true_ids, preds_ids, target_names=[str(idx_to_class[i]) for i in range(output_dim)]))


    plot_confusion_matrix(true_ids, preds_ids, target_names, prefix='nn_')

    #plot learning curve
    plot_learning_curve(epochs, train_loss_history,test_acc_history, prefix='nn_')

    #plot metrics bar chart
    plot_metrics_bar_chart(true_ids, preds_ids, target_names, prefix='nn_')


    return model, class_to_idx


def train_nn_with_focal_loss(model, train_loader, val_loader, num_epochs=10,
                          lr=1e-3, alpha=1.0, gamma=2.0,
                          use_scheduler=True, device="cpu"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, patience=3) if use_scheduler else None

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() 
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
       

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        

        if scheduler:
            scheduler.step(val_loss)

    idx_to_class = {i: str(i) for i in set(all_labels)}  # If your dataset has string labels, adjust accordingly
    target_names = [str(idx_to_class[i]) for i in sorted(idx_to_class.keys())]


    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    plot_confusion_matrix(all_labels, all_preds, target_names, prefix='nn_')
    plot_learning_curve(num_epochs, train_losses, val_accuracies, prefix='nn_')
    plot_metrics_bar_chart(all_labels, all_preds, target_names, prefix='nn_')
        
    return model

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