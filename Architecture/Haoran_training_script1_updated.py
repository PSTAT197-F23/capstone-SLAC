#!/usr/bin/env python
from argparse import ArgumentParser
parser = ArgumentParser()
#total_size = 1000  
#learning_rate = 0.003
# coding: utf-8
parser.add_argument("lr", type=float, default=0.03, help="learning rate (default=0.03)")
# In[3]:

parser.add_argument("size", type=int, default=5000, help="total size (default=5000)")

args = parser.parse_args()
learning_rate = args.lr
total_size = args.size
print("lr",learning_rate)
print("size",total_size)

# 1. Train the whole dataset (10400), 

# In[4]:

import h5py
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from pylab import *
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()


# In[5]:




# In[6]:


import torch.nn as nn

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.predictor = nn.Sequential(
            nn.Linear(128 * 99 * 99, 64),  
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.predictor(x)
        return x


# In[9]:


class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_layer6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 408 * 408, 10) # change to 10
        #self.fc1 = nn.Linear(64 * 202 * 202, 128)  
        #self.fc1 = nn.Linear(128 * 99 * 99, 256)  
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = x/255
        
        out = self.conv_layer1(x)
        #out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv_layer2(out)
        #out = self.bn1(out)
        out = self.relu1(out)
        out = self.max_pool1(out)

        '''
        out = self.conv_layer3(out)
        #out = self.bn2(out)
        out = self.relu1(out)
        out = self.conv_layer4(out)
        #out = self.bn2(out)
        out = self.relu1(out)
        out = self.max_pool2(out)

        
        out = self.conv_layer5(out)
        #out = self.bn3(out)
        out = self.relu1(out)
        out = self.conv_layer6(out)
        #out = self.bn3(out)
        out = self.relu1(out)
        out = self.max_pool3(out)
        '''
        out = out.view(out.size(0), -1)  
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        #out = torch.sigmoid(out) # smooth/sharp?
        
        return out


# In[ ]:


h = h5py.File("master.hdf5", "r")


# In[ ]:


class TrainingData(Dataset):
    
    def __init__(self, file, transform = None, target_transform = None, **kwargs):
        '''
        A class to initialize our training data.
        Args:
            file: string (master.hdf5)
            transform: callable function to apply to images
            target_transform: callable function to apply to target

        Initiate: data = TrainingData("master...")
        '''
        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.imgs, self.labels = self._extract_data()

    def __len__(self):
        '''
        Grabs the number of observations
        '''
        assert len(self.imgs) == len(self.labels)
        return len(self.imgs)

    def __getitem__(self, idx):
        '''
        this is how we can select examples
        '''
        img = self.imgs[idx].astype(np.float32)
        label = self.labels[idx].reshape(2,1).astype(np.float32)
        
        if self.transform:
            img = self.tranform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def _extract_data(self, **kwargs):
        '''
        Extracts the images and the labels from the hdf5 label.
        Returns: Tuple --> (images, labels)
        '''
        # open our master file 
        h = h5py.File(self.file, "r")
        # grab the images 
        imgs = h['images']
        # grab the labels 
          # first find the indexes with the attribbutes 
        idx1 = list(h['labels'].attrs['names']).index('cent_fast_train') # ground truth
        idx2 = list(h['labels'].attrs['names']).index('cent_slow_train') # absolute center
        # then we extract 
        labels = h['labels'][:, [idx1, idx2]]
    
        return imgs, labels


# In[ ]:


# Create an instance of the TrainingData class
training_data = TrainingData('master.hdf5')

# Create DataLoader instances for training and testing
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Fetch the first batch of data from the train_dataloader
train_features, train_labels = next(iter(train_dataloader))

# Display the batch size
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


# In[ ]:


train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

limited_training_data, _ = random_split(training_data, [total_size, len(training_data)-total_size])

train_dataset, val_dataset, test_dataset = random_split(limited_training_data, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

train_features, train_labels = next(iter(train_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


# In[7]:


model = ConvNeuralNet()
if use_gpu:
    model = model.cuda()
    
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  


# In[10]:


learning_rate = 0.003
model1 = ConvNeuralNet()
if use_gpu:
    model1 = model.cuda()
    
optimizer = torch.optim.SGD(model1.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  


# In[11]:


summary(model1, input_size=(1, 820, 820))


# In[8]:


summary(model, input_size=(1, 820, 820))


# In[ ]:


class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super(EuclideanDistanceLoss, self).__init__()

    def forward(self, outputs, labels):
        loss = torch.sqrt(torch.sum((outputs - labels) ** 2, dim=1))

        mean_loss = torch.mean(loss)

        return mean_loss
criterion = EuclideanDistanceLoss()


# In[ ]:


num_epochs = 1000

for epoch in range(num_epochs):
    model.train()  
    total_train_loss = 0
    
    # Training phase
    for images, labels in train_dataloader:
        images = images.to(device).unsqueeze(1)
        labels = labels.to(device).squeeze(-1) /820
        #print("Labels:",torch.mean(labels*820, 0))
        outputs = model(images)
        #print("Outputs:",torch.mean(outputs*820, 0))
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        #print(loss.item())
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device).unsqueeze(1)
            labels = labels.to(device).squeeze(-1) /820
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')


# In[ ]:


#torch.cuda.empty_cache()


# In[ ]:


model.eval()
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        images = images.unsqueeze(1)
        labels = labels.to(device)
        labels = labels.squeeze(-1)
        outputs = model(images)
        print("outputs", outputs*820)
        print("labels",labels)
        loss = criterion(outputs*820, labels)
        print(loss)
