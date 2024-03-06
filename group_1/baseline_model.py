from torchvision import models
from torch import nn
# we will implement a simple CNN to grab the data and keep this simple
class simpleCNN(nn.Module):
    def __init__(self, input_shape:int, output_shape:int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(
            in_channels = input_shape,
            out_channels = 32,
            stride = 1,
            padding = 'same',
            kernel_size = 3
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            stride = 1,
            padding = 'same',
            kernel_size = 3
        )
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            stride = 1,
            padding = 'same',
            kernel_size = 3
        )
        self.conv_f = nn.Conv2d(
            in_channels=128,
            out_channels=10,
            kernel_size = 1,
            padding = 0,
            stride = 1
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            in_features = 420250,
            out_features = 10
        )
        # this gives us a tensor of length ten 
        self.output = nn.Linear(
            in_features=10,
            out_features=output_shape
        )
        # this outputs the desired shape we have to reshape this to (2,1)
        self.relu = nn.ReLU()

    # we then pass this through the forward function 
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu((self.conv3(x)))
        x = self.flatten(self.conv_f(x))
        x = self.output(self.relu(self.fc1(x)))
        return x.unsqueeze(2)

        