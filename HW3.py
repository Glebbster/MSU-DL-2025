import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.nn import functional as F

import torch.utils.data.dataloader
from torchvision import datasets, transforms

# Load data
train_data = datasets.CIFAR10(root = './HW3_CIFAR10_data', train =True, download= True, transform = transforms.ToTensor())
test_data = datasets.CIFAR10(root = './HW3_CIFAR10_data', train = False, download= True, transform = transforms.ToTensor())

train_size = int(len(train_data)*0.8)
val_size = len(train_data) - train_size

train_data, val_data = torch.utils.data.random_split(train_data,[train_size,val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size= 64, shuffle = False)


def show_images(images, labels):
    f, axes= plt.subplots(1, 10, figsize=(30,5))

    for i, axis in enumerate(axes):
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0))

        axes[i].imshow(img)
        axes[i].set_title(labels[i].numpy())

    plt.show()

for batch in train_loader:
    images, labels = batch
    break

# show_images(images, labels)

print(images.shape)

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b_norm = nn.BatchNorm1d(num_features= 32*32*3)
        self.lin_1 = nn.Linear(in_features=32*32*3, out_features=1024, bias= True)
        self.b_norm1 = nn.BatchNorm1d(num_features= 1024)
        self.lin_2 = nn.Linear(in_features=1024, out_features=512, bias= True)
        self.b_norm2 = nn.BatchNorm1d(num_features= 512)
        self.lin_3 = nn.Linear(in_features=512, out_features=256, bias = True)
        self.b_norm3 = nn.BatchNorm1d(num_features= 256)
        self.out = nn.Linear(in_features=256, out_features=10, bias= False)

    def forward(self,x):     
        x = self.b_norm(x)

        x = F.relu(self.lin_1(x))
        x = self.b_norm1(x)
       
        x = F.relu(self.lin_2(x))
        x = self.b_norm2(x)
        
        x = F.relu(self.lin_3(x))
        x = self.b_norm3(x)

        x = self.out(x)
        return(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = NN().to(device)

# task assert
assert model is not None, 'Переменная model пустая. Где же тогда ваша модель?'

try:
    x = images.reshape(-1, 3072).to(device)
    y = labels

    # compute outputs given inputs, both are variables
    y_predicted = model(x)
except Exception as e:
    print('С моделью что-то не так')
    raise e


assert y_predicted.shape[-1] == 10, 'В последнем слое модели неверное количество нейронов'


loss_fn = nn.CrossEntropyLoss()
learn_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr = learn_rate)

# write func for train, validate
def validate(model,data_loader,loss_fn):
    losses = []
    y_pred = []
    full_y_pred = []
    full_y_true = []
    # make with progress bar
    for i,batch in enumerate(tqdm(data_loader)): 
        image, label = batch
        x = image.reshape(-1,32*32*3)
        # block grad changes
        with torch.no_grad():
            label_predicted = model(x.to(device))
            # get loss value
            loss = loss_fn(label_predicted, label.to(device))
            loss = loss.item()
            losses.append(loss)
            # get answers 
            y_pred = torch.argmax(label_predicted,dim = 1)
        # connect all model and val answers 
        full_y_pred.extend(y_pred.cpu().numpy())
        full_y_true.extend(label.cpu().numpy())
    accuracy = accuracy_score(full_y_pred,full_y_true)
    return accuracy, np.mean(losses)        


def train_epoch(model,data_loader,loss_fn,optimizer):
    for image, label in data_loader:
        x = image.reshape(-1, 32*32*3).to(device)
        label = label.to(device)

        
        label_predicted = model(x)
        loss = loss_fn(label_predicted, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('Epoch_train_acc,loss:',validate(model,data_loader,loss_fn))
    
    print('Epoch_val_acc,loss:',validate(model,val_loader,loss_fn))

for epoch in range(6):
    print('Epoch:',epoch)
    train_epoch(model,data_loader = train_loader, loss_fn= loss_fn, optimizer= optimizer)
    
print("Test acc:", validate(model,test_loader,loss_fn))

x = torch.randn((64, 32*32*3))
torch.jit.save(torch.jit.trace(model.cpu(), (x)), "model.pth")