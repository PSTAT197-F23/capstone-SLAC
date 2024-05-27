import torch 
from torch.utils.data import Subset, DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

def visualize_example(data, idx):
    '''
        Simple function to visualize just one example
        Parameters:
            data: An instance of the image class
            idx: which image you would like to look at
        Returns:
            A visual of the diffraction image
    '''
    plt.imshow(data[idx][0].squeeze(), vmax = 40, cmap = 'gray_r')
    # plot the center of the image 
    plt.scatter(data[idx][1][0].item(), data[idx][1][1].item(), marker = 'x', color = 'red')
    plt.title(f'The center is at ({data[idx][1][0].item():.3f}, {data[idx][1][1].item():.3f})')