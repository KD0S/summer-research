import numpy as np
from pandas import test

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import preprocessor
from model import BiLSMTNet
from model_bert import InCaseLawBERT
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.labels = y
        self.texts = x
        
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

        
batch_size = 8
preprocess = preprocessor.Preprocessing(0.2)
max_len = 512
preprocess.load_data()


x_train = preprocess.x_train
x_test = preprocess.x_test
y_train = preprocess.y_train
y_test = preprocess.y_test
x_val = preprocess.x_val
y_val = preprocess.y_val

_, x_val, _, y_val = train_test_split(x_val, y_val, shuffle=True, stratify=y_val,
                                        test_size=0.2)
model = BiLSMTNet(max_len).to('cuda')
# model = InCaseLawBERT().to('cuda')


from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import Recall


accuracy = Accuracy(task="binary", num_classes=2).to('cuda')
recall = Recall(task="binary", num_classes=2).to('cuda')
precision = Precision(task="binary", num_classes=2).to('cuda')


def train():
    # current_best : 0.60
    best_acc = 0.50
    trainings_set = Dataset(x_train, y_train)
    val_set = Dataset(x_val, y_val)
    train_loader = DataLoader(trainings_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        total_correct_train = 0
        total_samples_train = 0
        total_correct_test = 0
        total_samples_test = 0
        model.train()
        for data, target in tqdm(train_loader):
            
            y = target.to('cuda')
            optimizer.zero_grad()
        
            y_pred = model(data.to('cuda'))
            loss = criterion(y_pred.squeeze(), y.float())
            loss.backward()
            optimizer.step()
            predicted = torch.round(y_pred.squeeze()).type(torch.int64)
            total_correct_train += (predicted == y).sum().item()
            total_samples_train += y.size(0)
                        
        model.eval()
        with torch.no_grad():
            
            for data, target in tqdm(val_loader):
                
                y = target.to('cuda')
                y_pred = model(data.to('cuda'))
                predicted = torch.round(y_pred.squeeze()).type(torch.int64)
                total_correct_test += (predicted == y).sum().item()
                total_samples_test += y.size(0)

        train_accuracy = total_correct_train/total_samples_train
        test_accuracy = total_correct_test/total_samples_test
        
        if best_acc < test_accuracy:
            torch.save(model.state_dict(), './savedModels/BiLSTMmodel_no_aug.pt')
            best_acc = test_accuracy

        print("Epoch %d, loss: %.5f, Train accuracy: %.5f, Validation accuracy: %.5f" % (epoch+1, loss.item(), train_accuracy, test_accuracy))                 

def test():
    model = BiLSMTNet(max_len).to('cuda')
    model.load_state_dict(torch.load('./savedModels/BiLSTMmodel_no_aug.pt'))
    total_correct_test = 0
    total_samples_test = 0
    test_set = Dataset(x_test, y_test)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True)
    model.eval()
    with torch.no_grad():  
        for data, target in tqdm(test_loader):
            
            y = target.to('cuda')
            y_pred = model(data.to('cuda'))
            predicted = torch.round(y_pred.squeeze()).type(torch.int64)
            total_correct_test += (predicted == y).sum().item()
            total_samples_test += y.size(0)
    
    test_accuracy = total_correct_test/total_samples_test
    
    print("Test accuracy: %.5f" % (test_accuracy))
            
# train()
# test()

print(x_train)