import torch 
from torch.utils.data import Subset, DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

class ImageFolder(Dataset):
    # initialize with the directory path to master as well as the optional transform
    def __init__(self, target_dir:str, transform = None, target_transform = None):
        # initalize the parameters
        self.target_dir = target_dir
        self.transform = transform 
        self.target_transform = target_transform
        self.imgs, self.labels = self._extract_data()
        
    def _extract_data(self, **kwargs):
        '''
        Extracts the images and the labels from the hdf5 label.
        Returns: Tuple --> (images, labels)
        '''
        # open our master file 
        h = h5py.File(self.target_dir, "r")
        # grab the images 
        imgs = h['images']
        # grab the labels 
          # first find the indexes with the attribbutes 
        idx1 = list(h['labels'].attrs['names']).index('cent_fast_train')
        idx2 = list(h['labels'].attrs['names']).index('cent_slow_train')
        # then we extract 
        labels = h['labels'][:, [idx1, idx2]]
        return imgs, labels

    def __len__(self) -> int:
        # returns the number of training examples in dataset
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
            # convert to tensors 
            img = torch.from_numpy(img)
    
            # include the number of channels for the grey scale images
            img = img.unsqueeze(2)
        if self.target_transform:
            label = torch.from_numpy(label)
            label = self.target_transform(label)
        else:
            label = torch.from_numpy(label)


        return img, label


def create_validation(data, train_prop):
    '''
        This function will create a validation set by simple indexing.
        Parameters: 
            Data: The data should be of class ImageFolderCustom
            Train_prop: The percentage you want in training 
            
    '''
    # we first calculate how many observations we want in the training set
    training_size = round(len(data) * train_prop)

    # then we create a list of random indexes to ensure random sampling 
    indices = torch.randperm(len(data))

    # we then grab the training and validation indexes
    train_idxs = indices[:training_size]
    val_idxs = indices[training_size:]

    # we then subset the dataset to get both 
    train_data = Subset(data, train_idxs)
    val_data = Subset(data, val_idxs)

    return train_data, val_data


# now we need to create a function to create the dataloaders part
def create_dataloaders(training_dir:str, batch_size:int, num_workers:int,
                       testing_dir=None, transform=None, target_transform=None,
                       train_prop=0.8):
    '''
        This function will use all of the tools we created to create a training and 
        testing set. We will utilize the ImageFolderCustom class and our create validiation
        function to ease the process of loading the data. The parameter testing_dir
        will be implemented in the future for when Derek provides the testing data. 

        Parameters:
            training_dir: The directory to the training data 
            testing_dir: The directory to the testing
            transform: transformation to apply to the data 
            batch_size: The number of batches per dataloader
            num_workers: the number of cpus to be used when loading in the data
            train_prop: The proportion of training data to create the validation

        Returns: 
            A tuple of (train_dataloader, test_dataloader)

    '''

    # we first create the dataset 
    data = ImageFolder(training_dir)


    # we then call the create validation to create an aritifical testing set 
    training_set, testing_set = create_validation(data, train_prop)

    # we finally call the dataloaders
    train_dataloader = DataLoader(
        training_set,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers
    )
    
    test_dataloader = DataLoader(
        testing_set,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers
    )

    return train_dataloader, test_dataloader
    
