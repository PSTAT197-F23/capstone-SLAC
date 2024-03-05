#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
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

