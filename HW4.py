import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from torchvision import transforms, datasets
# load data
train_data = datasets.CIFAR10(root= './HW3_CIFAR10_data', train= True, transform= transforms.ToTensor())
test_data = datasets.CIFAR10(root= './HW3_CIFAR10_data', train= False, transform= transforms.ToTensor())

# create data, validation and test dataloaders
train_size = int(len(train_data)*0.8)
val_size = len(train_data) - train_size

train_data,val_data = torch.utils.data.random_split(train_data,[train_size,val_size])

train_loader = DataLoader(train_data,batch_size = 64, shuffle = True)
val_loader = DataLoader(val_data,batch_size = 64, shuffle = False)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False)
# show data
def show_images(images, labels):
    f, axes = plt.subplots(1,5)
    for i, axis in enumerate(axes):
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0))

        axes[i].imshow(img)
        axes[i].set_title(labels[i].numpy())
    plt.show()
for batch in train_loader:
    images, labels = batch
    break
# show_images(images,labels)
# create model
class My_model(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = 9)
        
        self.pull = nn.MaxPool2d(kernel_size = (2,2))
        self.flat = nn.Flatten()

        self.lin_1 = nn.Linear(in_features = 7*7*10, out_features = 64, bias = True)
        self.lin_2 = nn.Linear(in_features = 64, out_features = 32, bias = True)
        self.lin_3 = nn.Linear(in_features = 32, out_features = 10, bias = False)

        self.b_norm2 = nn.BatchNorm1d(64)
        self.b_norm3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pull(x)
        x = self.conv2(F.relu(x))

        x = self.flat(x)
        x = self.lin_1(x)
    
        x = self.b_norm2(F.relu(x))   
        x = self.lin_2(x)

        x = self.b_norm3(F.relu(x))
        x = self.lin_3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = My_model().to(device)
# count accuracy and loss function
def validate(model,dataloader,loss_fn):
    losses = []
    y_predicted_full = []
    y_true_full = []
    for i,batch in tqdm(enumerate(dataloader)):
        x, y_true = batch
        with torch.no_grad():
            logits = model(x.to(device))
            y_pred = torch.argmax(logits, dim = 1 )

            loss = loss_fn(logits, y_true.to(device))
            loss = loss.item()
            losses.append(loss)
            
        y_predicted_full.extend(y_pred.cpu().numpy())
        y_true_full.extend(y_true.cpu().numpy())
    accuracy = accuracy_score(y_predicted_full, y_true_full) 
    return accuracy, np.mean(losses)
#  train model function
def train_model(model,optimizer,loss_fn,dataloader, num_epochs = 6):
    for epoch in range(num_epochs):
        print('Epoch:', epoch)
        for i, batch in tqdm(enumerate(dataloader)):
            image, label = batch

            logits = model(image.to(device))
            y_true = label.to(device)

            # optimize parameters
            loss = loss_fn(logits,y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('train_accuracy,loss:', validate(model,dataloader,loss_fn))
        
        print('val_accuracy,loss:', validate(model,dataloader = val_loader,loss_fn= loss_fn))

# set up lossfn and optim
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

train_model(model, optimizer = optimizer, loss_fn = loss_fn, dataloader = train_loader, num_epochs = 7)
print('test_accuracy:', validate(model,test_loader,loss_fn))
# final accuracy score on test is 0.62, goal was accuracy above 0.6