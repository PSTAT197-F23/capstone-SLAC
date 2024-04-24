import torch
from torchvision import models
from torch import optim
from torchinfo import summary
from torch import nn



def create_resnet(device:torch.device, model_type:int, hidden_units:int, batch_size:int):
    '''
        This function creates the Residual Network architecture. And prints out the summary

        Parameters:
            device: The torch.device to send the model architecture to
            model_type: What type of residual network to use [18, 32, ..]
            hidden_units: How many activations to add before the output layer
            batch_size: The training batch size 

        Returns:
            The final model 

    '''
    # grab the resnet architecture from pytorch
    resnet_arch = getattr(models, "resnet%d" % model_type)
    resnet = resnet_arch().to(device)

    # grab the first convolutional layer
    rc = resnet.conv1

    # alter resnet.conv1 to have 1 input channl (other args are same
    resnet.conv1 = nn.Conv2d(
        1, rc.out_channels, kernel_size=rc.kernel_size, stride=rc.stride, 
        padding=rc.padding, bias=rc.bias, device=device
    )

    resnet_outputs = resnet.fc.out_features

    resnet_fc1 = nn.Linear(in_features=resnet_outputs,
                           out_features=hidden_units, device = device)
    resnet_fc2 = nn.Linear(hidden_units, 2, device = device)
    model = nn.Sequential(resnet, resnet_fc1, nn.ReLU(), resnet_fc2)

    # print out the summary
    print(summary(model=model, 
        input_size=(16, 1, 820, 820),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]))
    
    return model

    