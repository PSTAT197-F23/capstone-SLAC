import h5py
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from torch.optim import RMSprop
import torch.nn.functional as F
import matplotlib.pyplot as plt


h = h5py.File("master.hdf5", "r")
h['images']
h['labels']

idx1 = list(h['labels'].attrs['names']).index('pitch_deg') 
idx2 = list(h['labels'].attrs['names']).index('yaw_deg') 
labels = h['labels'][:, [idx1, idx2]]

# img = np.array(h['images'][0]).astype(np.float32)[None,None]
# print(img.shape)
# # should be 1,1,820,820 or whatever
# img_t = torch.tensor(img)
# exit()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("Using device", device)

#Dataloader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ImageData(Dataset):
    
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
        img = self.imgs[idx].astype("float32")
        label = self.labels[idx].astype("float32")
        
        if self.transform:
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)
            img = self.transform(img).permute(1,2,0)
        else:
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)
        
        if self.target_transform:
            label = torch.from_numpy(label)
            label = self.target_transform(label)
        else:
            label = torch.from_numpy(label)

        return img, label

    def _extract_data(self, **kwargs):
        '''
        Extracts the images and the labels from the hdf5 label.
        Returns: Tuple --> (images, labels)
        '''
        h = h5py.File(self.file, "r")
        imgs = h['images']
        idx1 = list(h['labels'].attrs['names']).index('pitch_deg')
        idx2 = list(h['labels'].attrs['names']).index('yaw_deg')
        labels = h['labels'][:, [idx1, idx2]]
    
        return imgs, labels

params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 1}

# Train dataset
train_dataset = ImageData('master.hdf5')
train_loader = DataLoader(dataset=train_dataset, **params)

# validation dataset
val_dataset = ImageData('master.hdf5')
val_loader = DataLoader(dataset=val_dataset, **params)

# test dataset
test_dataset = ImageData('master.hdf5')
test_loader = DataLoader(dataset=test_dataset, **params)

dataloaders_dict = {'train':train_loader, 'val':val_loader}

features, labels = next(iter(train_loader))
print(f'Train Features: {features.shape}\nTrain Labels: {labels.shape}')
print()
features, labels = next(iter(val_loader))
print(f'Validation Features: {features.shape}\nValidation Labels: {labels.shape}')
print()
# features = next(iter(test_loader))
# print(f'Test Features: {features.shape}\n')

#Resnet
# class baseBlock(torch.nn.Module):
#     expansion = 1
#     def __init__(self,input_planes,planes,stride=1,dim_change=None):
#         super(baseBlock,self).__init__()
#         #declare convolutional layers with batch norms
#         self.conv1 = torch.nn.Conv2d(input_planes,planes,stride=stride,kernel_size=3,padding=1)
#         self.bn1   = torch.nn.BatchNorm2d(planes)
#         self.conv2 = torch.nn.Conv2d(planes,planes,stride=1,kernel_size=3,padding=1)
#         self.bn2   = torch.nn.BatchNorm2d(planes)
#         self.dim_change = dim_change
#     def forward(self,x):
#         #Save the residue
#         res = x
#         output = F.relu(self.bn1(self.conv1(x)))
#         output = self.bn2(self.conv2(output))

#         if self.dim_change is not None:
#             res = self.dim_change(res)
        
#         output += res
#         output = F.relu(output)

#         return output

# class bottleNeck(torch.nn.Module):
#     expansion = 4
#     def __init__(self,input_planes,planes,stride=1,dim_change=None):
#         super(bottleNeck,self).__init__()

#         self.conv1 = torch.nn.Conv2d(input_planes,planes,kernel_size=1,stride=1)
#         self.bn1 = torch.nn.BatchNorm2d(planes)
#         self.conv2 = torch.nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1)
#         self.bn2 = torch.nn.BatchNorm2d(planes)
#         self.conv3 = torch.nn.Conv2d(planes,planes*self.expansion,kernel_size=1)
#         self.bn3 = torch.nn.BatchNorm2d(planes*self.expansion)
#         self.dim_change = dim_change
    
#     def forward(self,x):
#         res = x
        
#         output = F.relu(self.bn1(self.conv1(x)))
#         output = F.relu(self.bn2(self.conv2(output)))
#         output = self.bn3(self.conv3(output))

#         if self.dim_change is not None:
#             res = self.dim_change(res)
        
#         output += res
#         output = F.relu(output)
#         return output

# class ResNet(torch.nn.Module):
#     def __init__(self,block,num_layers,classes=1103):
#         super(ResNet,self).__init__()
#         self.input_planes = 16
#         self.conv1 = torch.nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1)
#         self.bn1 = torch.nn.BatchNorm2d(16)
#         self.layer1 = self._layer(block,16,num_layers[0],stride=1)
#         # self.layer2 = self._layer(block,128,num_layers[1],stride=2)
#         # self.layer3 = self._layer(block,256,num_layers[2],stride=2)
#         # self.layer4 = self._layer(block,512,num_layers[3],stride=2)
#         self.averagePool = torch.nn.AvgPool2d(kernel_size=4,stride=1)
#         self.fc    =  torch.nn.Linear(16*block.expansion,classes)
    
#     def _layer(self,block,planes,num_layers,stride=1):
#         dim_change = None
#         if stride!=1 or planes != self.input_planes*block.expansion:
#             dim_change = torch.nn.Sequential(torch.nn.Conv2d(self.input_planes,planes*block.expansion,kernel_size=1,stride=stride),
#                                              torch.nn.BatchNorm2d(planes*block.expansion))
#         netLayers =[]
#         netLayers.append(block(self.input_planes,planes,stride=stride,dim_change=dim_change))
#         self.input_planes = planes * block.expansion
#         for i in range(1,num_layers):
#             netLayers.append(block(self.input_planes,planes))
#             self.input_planes = planes * block.expansion
        
#         return torch.nn.Sequential(*netLayers)

#     def forward(self,x):
#         x = F.relu(self.bn1(self.conv1(x)))

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = F.avg_pool2d(x,4)
#         x = x.view(x.size(0),-1)
#         x = self.fc(x)

#         return x

# NeuralNet  =  ResNet(bottleNeck,[3,4,6,3])
# NeuralNet.to(device)
# summary(NeuralNet, (1, 820, 820))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(SimpleResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# Define the ResNet architecture
def simple_resnet():
    return SimpleResNet(ResidualBlock, [2, 2, 2, 2])

# Create an instance of the model
model = simple_resnet()
model.to(device)
summary(model, (1, 820, 820))

# #Train
# x_dim = 820
# y_dim = 820
# hidden_dim = 400
# latent_dim = 200
# batch_size = 8
# rgb = 1
# lr = 1e-20
# epochs = 30

# L1_loss = nn.L1Loss()

# def loss_function(x, x_hat, mean, log_var):
#     reproduction_loss = L1_loss(x_hat, x)
#     KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

#     return reproduction_loss + KLD

# optimizer = RMSprop(model.parameters(), lr=lr)

# print("Start training ResNet...")
# model.train()

# epoch_losses = []
# num_images = 64
# num_batches = -(-num_images // batch_size)  # Ceiling division to get the number of batches needed

# for epoch in range(epochs):
#     overall_loss = 0
#     num_processed_images = 0  
    
#     for batch_idx, (features, labels) in enumerate(train_loader):
#         if num_processed_images >= num_images:
#             break
        
#         x = features.view(batch_size, rgb, x_dim, y_dim)
#         x = x.to(device)

#         optimizer.zero_grad()

#         x_hat, mean, log_var = model(x)
#         loss = loss_function(x, x_hat, mean, log_var)
        
#         overall_loss += loss.item()
        
#         loss.backward()
#         optimizer.step()
        
#         num_processed_images += len(features)
        
#     epoch_loss = overall_loss / num_images  # Compute average loss per image
#     epoch_losses.append(epoch_loss)
    
#     print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", epoch_loss)
    
# print("Finish!!")

import torch
import torch.nn as nn
import torch.optim as optim

# Assuming model, train_loader, val_loader, and device are already defined

criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.005, momentum=0.9)

num_epochs = 100

for epoch in range(num_epochs):
    # Set model to train mode
    model.train()
    
    running_loss = 0.0
    
    # Iterate over the training dataset
    for batch_idx, (features, labels) in enumerate(train_loader):
        # Move data to device
        features, labels = features.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if batch_idx % 100 == 99:    # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / (batch_idx + 1)))
            running_loss = 0.0
    
    # Validation
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()

    # Print validation loss
    print('Epoch [%d/%d], Validation Loss: %.4f' % (epoch + 1, num_epochs, val_loss / len(val_loader)))

print('Finished Training')


# Plotting the training and validation losses
plt.plot(running_loss / len(train_loader), label='Training Loss')
plt.plot(val_loss / len(val_loader), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()
