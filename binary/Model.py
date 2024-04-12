#!/usr/bin/env python
# coding: utf-8

# ### Load Dataset

# In[5]:


import torch
print(torch.__version__)


# In[9]:



import h5py

h = h5py.File("master.hdf5", "r")


# In[2]:


h['images']
#<HDF5 dataset "images": shape (10240, 832, 832), type "<u2">


# In[3]:


h['labels']
#<HDF5 dataset "labels": shape (10240, 30), type "<f4">


import matplotlib.pyplot as plt
import numpy as np
from pylab import *


# In[27]:


# import matplotlib.pyplot as plt

# idx1 = list(h['labels'].attrs['names']).index('cent_fast_train') #ground truth
# idx2 = list(h['labels'].attrs['names']).index('cent_slow_train') #absolute center
# labels = h['labels'][:, [idx1, idx2]]

# for i in range(10):
#     center_x, center_y = labels[i]
#     fig, ax = plt.subplots()
#     ax.imshow(h['images'][i], vmax=40, cmap='gray_r')
#     ax.scatter(center_x, center_y, color='red', marker='o')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.show()


# In[5]:


import matplotlib.pyplot as plt

idx1 = list(h['labels'].attrs['names']).index('cent_fast_train') #ground truth
idx2 = list(h['labels'].attrs['names']).index('cent_slow_train') #absolute center
labels = h['labels'][:, [idx1, idx2]]

# fig, axes = plt.subplots(10, 10, figsize=(15, 15))

# for i in range(10):
#     for j in range(10):
#         index = i * 10 + j
#         axes[i, j].imshow(h['images'][index], vmax=40, cmap='gray_r')
#         axes[i, j].axis('off')
#         axes[i, j].add_patch(plt.Circle(xy=labels[index], radius=200, fc='none', ec='r'))

# plt.show()


# ### Data Loader

# In[6]:


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
        idx1 = list(h['labels'].attrs['names']).index('cent_fast_train')
        idx2 = list(h['labels'].attrs['names']).index('cent_slow_train')
        # then we extract 
        labels = h['labels'][:, [idx1, idx2]]
    
        return imgs, labels


# In[7]:


# Create an instance of the TrainingData class
training_data = TrainingData('master.hdf5')

# Create DataLoader instances for training and testing
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Fetch the first batch of data from the train_dataloader
train_features, train_labels = next(iter(train_dataloader))

# Display the batch size
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")


# In[8]:


# import matplotlib.pyplot as plt

# image1 = train_features[0]  
# image2 = train_features[1] 

# image1 = image1.numpy()
# image2 = image2.numpy()


# plt.figure(figsize=(5, 5))
# plt.imshow(image1.squeeze(), cmap='gray_r')
# plt.title('Image 1')
# plt.axis('off')
# plt.show()

# plt.figure(figsize=(5, 5))
# plt.imshow(image2.squeeze(), cmap='gray_r')
# plt.title('Image 2')
# plt.axis('off')
# plt.show()


# ### Dimension Reduction: VAE

# In[9]:


x_dim = 820
y_dim = 820
hidden_dim = 400
latent_dim = 200
batch_size = 64
lr = 1e-20
epochs = 30


# In[10]:


import torch
import torch.nn as nn


# In[11]:


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("Using device", device)

# In[12]:


"""
    A simple implementation of Gaussian MLP Encoder and Decoder
"""

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var


# In[13]:


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


# In[14]:


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var


# In[15]:


encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = Model(Encoder=encoder, Decoder=decoder).to(device)


# In[16]:


from torchsummary import summary

summary(model, input_size=(64, 820, 820))


# In[17]:


from torch.optim import RMSprop

L1_loss = nn.L1Loss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = L1_loss(x_hat, x)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

optimizer = RMSprop(model.parameters(), lr=lr)


# In[18]:


import matplotlib.pyplot as plt

print("Start training VAE...")
model.train()

epoch_losses = []
num_images = 50
num_batches = -(-num_images // batch_size)  # Ceiling division to get the number of batches needed

for epoch in range(epochs):
    overall_loss = 0
    num_processed_images = 0  
    
    for batch_idx, (features, labels) in enumerate(train_dataloader):
        if num_processed_images >= num_images:
            break
        
        x = features.view(batch_size, x_dim, y_dim)
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        num_processed_images += len(features)
        
    epoch_loss = overall_loss / num_images  # Compute average loss per image
    epoch_losses.append(epoch_loss)
    
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", epoch_loss)
    
print("Finish!!")

exit()

# In[19]:


# print("Start training VAE...")
# model.train()

# for epoch in range(epochs):
#     overall_loss = 0
#     for batch_idx, (features, labels) in enumerate(train_dataloader):
#         x = train_features.view(batch_size, x_dim, y_dim)
#         x = x.to(device)
#         print(x)

#         optimizer.zero_grad()

#         x_hat, mean, log_var = model(x)
#         loss = loss_function(x, x_hat, mean, log_var)
        
#         overall_loss += loss.item()
        
#         loss.backward()
#         optimizer.step()
        
#     print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    
# print("Finish!!")


# In[20]:


from tqdm import tqdm
from torchvision.utils import save_image, make_grid

model.eval()

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(train_dataloader)):
        x = x.view(batch_size, x_dim, y_dim)
        x = x.to(device)
        
        x_hat, _, _ = model(x)


# In[23]:


# def show_image(x, idx, batch_size=64):
#     x = x.view(batch_size, 820, 820)

#     fig = plt.figure()
#     plt.imshow(x[idx].cpu().numpy(), cmap='gray_r')
#     plt.show()


# # In[24]:


# show_image(x, idx=0)


# # In[25]:


# show_image(x_hat, idx=0)


# ### CNN Model

# In[31]:


import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(64, 2)  # Assuming 2 output classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = CNNModel()

print(model)


# In[32]:


criterion = nn.L1Loss()
optimizer = optim.RMSprop(model.parameters())

# Assuming train_features is the input feature batch
# Reshape train_features to match the expected input shape of the CNN
train_features = train_features.view(-1, 1, 820, 820)

# Training loop
epochs = 50
batch_size = 64
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_features)
    loss = criterion(outputs, train_labels.squeeze(1).long())
    loss.backward()
    optimizer.step()

    # Print statistics
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print('Training finished')


# In[ ]:




