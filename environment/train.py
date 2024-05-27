from argparse import ArgumentParser
import torch
from torch.utils.data import Subset, DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from data_setup import ImageFolder, create_validation, create_dataloaders
import h5py
import os
from torchvision import models 
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm.auto import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from resnet_arch import create_resnet
from engine import train_step, test_step, create_writer, train
from utils import save_model

# we create our parameters for training the file using the argpass module
# we first make a parser 
parser = ArgumentParser()

###########################   DATA LOADER ARGUMENTS #############################################

# parameter for the data file to extract from (master.hdf5)
parser.add_argument("--data_file", type=str, help = "training file for the data (hdf5)")

# parameter for the training batch size 
parser.add_argument("--train_bs", type=int, help = "Training batch size")

# parameter for the testing batch size 
parser.add_argument("--test_bs", type=int, help = "Testing batch size")

# parameter for the training set size. This is for a proportion
parser.add_argument("--train_size", type=float, help = "The proportion of the data for training")

# parameter for determining subgroup data. 
parser.add_argument("--tilt", action='store_true', help = "Nothing = cent data, False = tilt data") 

# parameter for the image transformation if image_transform is called 
parser.add_argument("--image_function", type=str, default = None, help = "The function used to transform the image data")

# parameter for the target transformation if target transform is called
parser.add_argument("--target_function", type=str, default = None,help = "The function used to transform the target coordinates")

############################# RESNET ARCHITECTURE PARAMETERS ######################################
# parameter for selecting the exact resnet architecture
parser.add_argument("--model_type", type=int, help="What Resnet model you want to try [18,32...]")

# parameter for determining the number of hiddden units before the output layer
parser.add_argument("--activations", type=int, help="Number of hidden units to add before the output")

############################  OPTIMIZER PARAMETERS ################################################
# parameter for the learning rate 
parser.add_argument("--lr", type=float, help="Learning rate")

# parameter for weight decay
parser.add_argument("--wd", type=float, help="weight decay")

# parameter for momentum
parser.add_argument("--mom", type=float, help="weight decay")

# parameter for optimizer 
parser.add_argument("--optimizer", type=str, help="Pytorch optimizer as a string")

################## TRAINING LOOP PARAMETERS #######################################################
# parameter for the loss 
parser.add_argument("--cost", type=str, help="PyTorch cost function")

# parameter for epochs
parser.add_argument("--epochs", type=int, help="Number of epochs")

# parameter for whether to normalize the target data 
parser.add_argument("--norm", action="store_true", help="If included normalize the target")

# parameter for scale normalization 
parser.add_argument("--scale", type=int, default=1, help="What to scale the target by if used")

############## EXPIREMENT PARAMETERS ##############################################################
# paramter for name of expirement
parser.add_argument("--expirement_name", type=str, help="Expirement Name")

# parameter for the model name 
parser.add_argument("--model_name", type=str, help="Name of the model")

# parameter for hyperparameters
parser.add_argument("--hyperparams", type=str, help="Hyperparameters")

############# SAVING THE MODEL PARAMERS #####################################################
# parameter for target directory
parser.add_argument("--target_dir", type=str, help="The folder that you want to save the model in")

# parameter for model name 
parser.add_argument("--model_save", type=str, help="Must end in .pth")



# collect all of the arguments 
args = parser.parse_args()

# we check to see if there is a gpu available and then we print this out for logging purposses 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#### GETTING THE DATALOADERS ####################################################################

data = args.data_file
training_batch_size = args.train_bs
testing_batch_size = args.test_bs
train_proportion = args.train_size
# if the tranformation is none then no eval
if args.image_function == None:
    image_function = args.image_function
else:
    image_function = eval(args.image_function)
if args.target_function == None:
    target_function = args.target_function
else:
    target_function = eval(args.target_function)


training_dataloader, testing_dataloader = create_dataloaders(training_dir=data,
                                                            train_batch_size=training_batch_size,
                                                            test_batch_size=testing_batch_size,
                                                            num_workers=os.cpu_count(),
                                                            train_prop=train_proportion,
                                                            transform=image_function,
                                                            target_transform=target_function,
                                                            cent= not args.tilt
                                                            )

##################### GETTING THE RESNET ARCHITECTURE ############################################
model_type = args.model_type
hidden_units = args.activations

# get the model and print the summary 
model = create_resnet(device=device, model_type=model_type, 
                      hidden_units=hidden_units,batch_size=training_batch_size)

#################### SETTING UP THE OPTIMIZER #####################################################
learning_rate = args.lr
weight_decay = args.wd
momentum = args.mom
optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

print(f"The optimizer used is {optimizer}")
#################### SETTING UP THE TRAINING LOOP #################################################
loss = getattr(nn, args.cost)()
print(f"The loss used is {loss}")
epochs = args.epochs
nbatch = len(training_dataloader)
scale = args.scale
expirement_name = args.expirement_name
model_name = args.model_name
hyperparameters = args.hyperparams

writer = create_writer(expirement_name=expirement_name,
                      model_name=model_name,
                      extra=hyperparameters)

# we will have two cases. One with normalization and one without normalization
if args.norm:
    results = train(model=model,
                   train_dataloader=training_dataloader,
                   test_dataloader=testing_dataloader,
                   optimizer=optimizer,
                   criterion=loss,
                   epochs=epochs,
                   device=device,
                   writer=writer,
                   nbatch=nbatch,
                   normalize=args.norm,
                   scale=scale)
else:
    results = train(model=model,
                   train_dataloader=training_dataloader,
                   test_dataloader=testing_dataloader,
                   optimizer=optimizer,
                   criterion=loss,
                   epochs=epochs,
                   device=device,
                   writer=writer,
                   nbatch=nbatch,
                   normalize=False,
                   scale=1)

#################################### SAVING THE MODEL #############################################
target_dir = args.target_dir
model_save = args.model_save

# we finally save the model
save_model(model=model, target_dir=target_dir, model_name=model_save)








