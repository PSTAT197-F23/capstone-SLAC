import h5py
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from torch.optim import RMSprop

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
        '''
        this is how we can select examples
        '''
        img = self.imgs[idx].astype("float32")
        label = self.labels[idx].reshape(2,1).astype("float32")
        
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


# training_data = TrainingData('master.hdf5')
# train_dataloader = DataLoader(training_data, **params)
# for batch in train_dataloader:
#     features, labels = batch[0], batch[1]
#     # print(batch[0])
#     print(features.shape, labels.shape)
#     break

# exit()
# train_features, train_labels = next(iter(train_dataloader))

# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")


#CNN
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
  def __init__(self, input_size=820):
    super(Net, self).__init__()
    resnet = models.resnet34(num_classes=1000)
    # resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze()) 
    rc = resnet.conv1
    resnet.conv1 = nn.Conv2d(
        1, rc.out_channels, kernel_size=rc.kernel_size, stride=rc.stride, 
        padding=rc.padding, bias=rc.bias, device=device
    )
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
    RESNET_FEATURE_SIZE = 128

    self.upsample = nn.Sequential(     
      nn.Conv2d(RESNET_FEATURE_SIZE, 128, stride=1, padding=1, kernel_size=3),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, stride=1, padding=1, kernel_size=3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, stride=1, padding=1, kernel_size=3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, stride=1, padding=1, kernel_size=3),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 16, stride=1, padding=1, kernel_size=3),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Conv2d(16, 2, stride=1, padding=1, kernel_size=3), 
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):
    midlevel_features = self.midlevel_resnet(input)
    output = self.upsample(midlevel_features)
    return output

model = Net().to(device)
summary(model, input_size=(1, 820, 820))

# x = torch.randn(8, 1, 820, 820).to(device)
# out = model(x)

#Train
x_dim = 820
y_dim = 820
hidden_dim = 400
latent_dim = 200
batch_size = 8
learning_rate = 1e-20
epochs = 30

criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

print("Start training CNN...")
model.train()

epoch_losses = []
num_images = 64
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


