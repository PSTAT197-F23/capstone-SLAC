from resnet_arch import create_resnet
import torch
import h5py
import os
import torch
import numpy as np


#define model architecture
model = create_resnet('cpu', 18, 200, 0)

#load model parameters
model.load_state_dict(torch.load('/home/sichenzhong/models/tilt2_firstmod.pth'))

#set model to eval mode
model.eval()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

from torch.utils.data import Dataset
class TrainingData(Dataset):
    
    def __init__(self, file, transform = None, target_transform = None, **kwags):
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
        img = torch.tensor(self.imgs[idx].astype(np.float32))
        label = torch.tensor(self.labels[idx].reshape(2,1).astype(np.float32))
        
        if self.transform:
            img = self.transform(img)
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
        idx1 = list(h['labels'].attrs['names']).index('pitch_deg')
        idx2 = list(h['labels'].attrs['names']).index('yaw_deg')
        # then we extract 
        labels = h['labels'][:, [idx1, idx2]]
    
        return imgs, labels

# pretrained model
model.eval()
model = model.to('cpu')
preds_random_tilt2 = []

from torch.utils.data import DataLoader
data = TrainingData('/home/sichenzhong/dials/binary/tilting.2/master.h5')
train_dataloader = DataLoader(data, batch_size=1)

with torch.no_grad():
    for X, y in train_dataloader:
        pred = model(X.unsqueeze(1)).unsqueeze(2).reshape(2)
        preds_random_tilt2.append(pred)
        y = y.reshape(2)-410
        print(f'ACTUAL: {y}   PREDICTED: {pred}')