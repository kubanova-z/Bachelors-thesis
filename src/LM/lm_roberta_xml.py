from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset


"""  Model initialization """

device = "cuda" if torch.cuda.is_available() else "cpu" # GPU, ked je dostupne, inak CPU

model_name = "xlm-roberta-base"
NUM_CLASSES = 4

tokenizer = AutoTokenizer.from_pretrained(model_name)

# nacitanie predtrenovaneho modelu xml-Roberta
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_CLASSES,  ignore_mismatched_sizes=True )
model.to(device)


""" Dataset class  """

class TextDataset(Dataset):
    # PyTorch dataset
    # premena textu na tokenizovane tensori
    def __init__(self, texts, labels, tokenizer, max_length = 300):
       self.texts = texts # vstupny text
       self.labels = labels
       self.tokenizer = tokenizer   # predtrenovany tokenizer
       self.max_length = max_length     # maximalna dlzka vzorky

    def __len__(self):
        return len(self.texts)  # pocet vzoriek v datasete
    
    def __getitem__(self, index):   # jedna vzorka
       text = self.texts[index]
       label = self.labels[index]
       encoding = self.tokenizer(
           text,
           truncation = True,
           padding = 'max_length',
           max_length = self.max_length,
           return_tensors = 'pt'
       )
       encoding = {k: v.squeeze(0) for k, v in encoding.items()}    # odstranenie batch dimenzie
       encoding['labels']= torch.tensor(label, dtype=torch.long)    # pridanie tensor labelu
       return encoding
    

""" Dataloader """
# konverzia textu a labelov na PyTorch Dataloaders objekty
# batchovanie, rozdelenie na test / train
def create_dataloaders(X_train, y_train, X_test, y_test, batch_size = 16):

    # trenovaci a testovaci dataset
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)

    # dataloadres - vytvorenie batchov, premiesanie dat pre kazdu epochu
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)
    return train_loader, test_loader


""" Training """
def train_model(model, train_loader, epochs = 3, lr = 2e-5):

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    # model v trenovacom mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k : v.to(device) for k, v in batch.items()}    # presunut a batche na GPU / CPU
            # forward
            outputs = model(**batch)    
            loss = outputs.loss
            # spatne sirenie chyby
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
    preds, labels = [], []

    with torch.no_grad(): # vypnutie vypoctu gradientu (rychlejsie) - uzitocne pocas trenovania
        for batch in test_loader:
            input_batch = {k : v.to(device) for k , v in batch.items() if k != 'labels'} # presun tensorov na GPU / CPU, bez labels
            outputs = model(**input_batch) # dictionary (vystuone skore pre kazdu triedu)
            preds_batch = outputs.logits.argmax(dim=1).cpu().numpy()    # vybratie indexu s najvacsou pravdepodobnostou pre kazdu triedu
            preds.extend(preds_batch)
            labels.extend(batch['labels'].numpy())

    from sklearn.metrics import classification_report
    print(classification_report(labels, preds, target_names=label_names))



""" Single text prediction """

def predict_sentence(text, model, tokenizer, target_names, device = 'cuda', max_length = 128):
    model.to(device)
    model.eval() # vyhodnocovaci rezim

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