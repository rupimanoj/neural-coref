import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from sklearn import datasets

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 8)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(8, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.relu2(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y
    
    
def train_epoch(model, opt, criterion, batch_size, X_data, Y_data, mode="train"):
    
    if(mode == "train"):
        model.train()
    else:
        model.eval()
    
    losses = []
    running_corrects = 0
    shuffled_idx = list(np.random.permutation(X_data.size()[0]))
    X_data_shuffled = [X_data[i] for i in shuffled_idx]
    Y_data_shuffled = [Y_data[i] for i in shuffled_idx]
    minibatch_idxs = np.array_split(shuffled_idx, len(shuffled_idx)/batch_size)
    for minibatch_ids in minibatch_idxs:
        x_batch = X_data[minibatch_ids]
        y_batch = Y_data[minibatch_ids]
        x_batch = Variable(x_batch).to(device)
        y_batch = Variable(y_batch).to(device)
        opt.zero_grad()
        
        if(mode == "train"):
            y_hat = model(x_batch)
        else:
            with torch.no_grad():
                y_hat = model(x_batch)
        
        y_preds = (y_hat > 0.5).type(torch.float32)
        loss = criterion(y_hat, y_batch)
        
        corrects = float(torch.sum(y_preds == y_batch).item())
        running_corrects += corrects
        
        if(mode == "train"):
            loss.backward()
            opt.step()
            
        losses.append(loss.item())
        
    accuracy = running_corrects * 1.0 / len(shuffled_idx)
    avg_loss = sum(losses) * 1.0 / len(losses)
    return avg_loss, accuracy