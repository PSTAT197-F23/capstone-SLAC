# create the training and testing step as functions to be called later in our file 
import torch 
from tqdm.auto import tqdm
from torch import optim
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


# the method we will use for the train step
def train_step(model: torch.nn.Module, 
              dataloader:torch.utils.data.dataloader.DataLoader,
              loss_fn:torch.nn.modules.loss,
               other_metric:torch.nn.modules.loss,
              optimizer:torch.optim.Optimizer,
              device:torch.device):
    '''
        Trains a PyTorch Model for a single epoch.

        Args: 
            1.) Model: The model that is desried to be used 
            2.) dataloader: The training dataloader to be used 
            3.) loss_fn: The loss function to be minimized 
            4.) optimizer: What flavour of gradient descent that we want to run 
            5.) device: The torch device (cuda)

        Returns:
            The tuple: (train_mse, train_mae)
    '''
    # we first put the model into train mode
    model.train()

    # set up the metrics that we want to track
    train_mse, train_mae= 0,0

    # we need to loop through our batches in the specified dataloader
    for batch, (X,y) in enumerate(dataloader):
        # we keep track of what batch we are on in the specific epoch 
        if batch % 100 == 0:
            print(f'On batch number {batch + 1}')

        # we permute the shape of our batch to match the content of the number of channels 
        X = X.permute(0,3,1,2)

        # we send our data to the correct device
        X, y = X.to(device), y.to(device)

        ## FORWARD PASS
        y_pred = model(X)

        # Calculate the loss and accumulate this for each batch
        loss_mse = loss_fn(y_pred, y)
        train_mse += loss_mse.item()

        # We also want to keep track of the MAE 
        loss_mae = other_metric(y_pred, y)
        train_mae += loss_mae.item()

        ## BACKPROPOGATION
        # we set the gradients to 0
        optimizer.zero_grad()

        # we calculate the derivatives 
        loss_mse.backward()

        # we use gradient clipping to help with exploding gradients 
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # we update the parameters
        optimizer.step()
        
    # we adjust our accumulative loss to get the loss per batch 
    train_mse /= len(dataloader)
    train_mae /= len(dataloader)
    return train_mse, train_mae

# the method for the testing step 
def test_step(model:torch.nn.Module,
             dataloader:torch.utils.data.dataloader.DataLoader,
             loss_fn:torch.nn.modules.loss,
             other_metric:torch.nn.modules.loss,
             device:torch.device):
    '''
        Tests a PyTorch Model for a single epoch

        Args:
            1.) model: A PyTorch model
            2.) dataloader: The testing dataloader
            3.) The loss function that will be used for gradient descent
            4.) Any other metric that is desrired to track 
            5.) The device we want to cache the tensors into 

        Returns: 
            A tuple: (test_mse, test_mae)
    '''

    # we first put the model into eval mode
    model.eval()

    # we then set up our metrics 
    test_mse, test_mae = 0,0

    # we turn on the inference context manger 
    with torch.inference_mode():
        # loop through the testing batches of the dataloader
        for X,y in dataloader:
            # premute X to match our model 
            X = X.permute(0,3,1,2)

            # then we send the data to the gpu
            X,y = X.to(device), y.to(device)

            ## FOWARD PASS
            test_pred = model(X)

            # compute the test lossese and accumulate
            test_mse += loss_fn(test_pred, y).item()
            test_mae += other_metric(test_pred, y).item()

    # get the loss per batch for both metrics 
    test_mse /= len(dataloader)
    test_mae /= len(dataloader)

    return test_mse, test_mae


# a helper function to create the summary writer 
def create_writer(expirement_name:str,
                 model_name:str,
                 extra:str=None):
    '''
        Creates a torch.utils.tensorboard.writer.SummaryWriter() to a specific directory

        Args:
            expirement_name: The name we want to call the expirement
            model_name: The name of the model
            extra: Anything extra to add to the file path 
            
    '''

    # grab the timestamp of the current date
    # we will write this function to keep all models that run on the same day in a folder

    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("runs", timestamp, expirement_name, model_name, extra)

    else: 
        log_dir = os.path.join("runs", timestamp, expirement_name, model_name)

    return SummaryWriter(log_dir=log_dir)


# now we need to write the overall train function 
def train(model:torch.nn.Module,
         train_dataloader:torch.utils.data.dataloader.DataLoader,
         test_dataloader:torch.utils.data.dataloader.DataLoader,
         optimizer:torch.optim.Optimizer,
         loss_fn:torch.nn.modules.loss,
         other_metric:torch.nn.modules.loss,
         epochs:int,
         device:torch.device,
         writer:torch.utils.tensorboard.writer.SummaryWriter):
    '''
        Trains and Tests a PyTorch model 
        Goes through the train_step and the test step 

        Returns: 
            A dictionary of training and testing loss as well as training and 
            testing metric. 
            In the form {train_loss: [epoch1, epoch2,...,epochn],
                        train_metric: [...],
                        test_loss: [...],
                        test_metric: [...]}
                        
    '''
    # first create an empty dictionary with the results that we want to keep 

    results = {
        "train_loss": [],
        "train_metric": [],
        "test_loss": [],
        "test_metric": []
    }

    # we loop through the training and testing steps 
    for epoch in tqdm(range(epochs)):
        # run through the training step by using the function
        train_loss, train_metric = train_step(model=model,
                                             dataloader=train_dataloader,
                                             loss_fn=loss_fn,
                                             other_metric=other_metric,
                                             optimizer=optimizer,
                                             device=device)
        # run through the corresponding testing step
        test_loss, test_metric = test_step(model=model,
                                          dataloader=test_dataloader,
                                          loss_fn=loss_fn,
                                          other_metric=other_metric,
                                          device=device)
        # print out where we are in training the model 
        print(
            f"EPOCH: {epoch + 1} | "
            f"TRAIN LOSS: {train_loss:.3f} | "
            f"TRAIN METRIC: {train_metric:.3f} | "
            f"TEST LOSS: {test_loss:.3f} | "
            f"TEST METRIC: {test_metric:.3f}"
        )

        # update our dictionary
        results['train_loss'].append(train_loss)
        results['train_metric'].append(train_metric)
        results['test_loss'].append(test_loss)
        results['test_metric'].append(test_metric)

        # EXPIREMENT TRACKING:
        # add loss results to the SummaryWriter
        writer.add_scalars(
            main_tag="LOSS MSE",
            tag_scalar_dict={"train_mse":train_loss,
                            "test_mse":test_loss},
            global_step=epoch
        )

        # add MAE to the SummaryWriter
        writer.add_scalars(
            main_tag='MAE',
            tag_scalar_dict={"train_mae":train_metric,
                            "test_mae":test_metric},
            global_step=epoch
        )
        
    writer.add_graph(
        model=model,
        input_to_model=torch.randn(32,1,820,820).to(device)
    )

    # we close the writer instance 
    writer.close()

    return results