import torch 
from tqdm.auto import tqdm
from torch import optim
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def train_step(model:torch.nn.Module, 
              dataloader:torch.utils.data.dataloader.DataLoader,
              criterion:torch.nn.modules.loss,
              optimizer:torch.optim.Optimizer,
              device:torch.device,
              scale:float,
              nbatch:int,
              normalize):

    # we first put the model into training mode
    model.train()
    
    train_mae = 0
    for i_batch, (images,labels) in enumerate(dataloader):
        # we keep track of what batch we are on in the specific epoch 
        if i_batch % 100 == 0:
            print(f'On batch number {i_batch + 1}')
        # apply normalization if necessary
        if normalize:
            labels /= scale
    
        # change the dimensions of the image to match the arch
        images = images.permute(0,3,1,2)
        labels = labels.squeeze()
    
        # send the tensors to the correct device
        images, labels = images.to(device), labels.to(device)
    
        # we set the optimizer gradients to 0
        optimizer.zero_grad()
    
        # grab the predictions of the model 
        preds = model(images)
    
        # we seperate the pres into seperate coordinates
        x, y = preds.T
    
        # Calculate the loss and accumulate this for each batch
        loss_mae = criterion(preds, labels)
        
        # we calculate the derivatives 
        loss_mae.backward()
    
        # we use gradient clipping to help with exploding gradients 
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # we update the parameters
        optimizer.step()
    
        # we add the total mae 
        train_mae += loss_mae.item()
    
        # calculate the standard deviation
        # xstd = x.std().item()
        # ystd = y.std().item()
    
        # configure print statements
        end = "\n" if i_batch==nbatch-1 else "\r"
        print(
            f"batch {i_batch+1}/{nbatch}: loss={loss_mae.item()*scale:.2f}", end=end, flush=True
        )
    
    train_mae /= len(dataloader)
    return train_mae


# the method for the testing step 
def test_step(model:torch.nn.Module,
             dataloader:torch.utils.data.dataloader.DataLoader,
             criterion:torch.nn.modules.loss,
             device:torch.device,
             scale:float,
             normalize):
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
    test_mae = 0

    # we turn on the inference context manger 
    with torch.inference_mode():
        
        # loop through the testing batches of the dataloader
        for X,y in dataloader:
            # premute X to match our model 
            X = X.permute(0,3,1,2)
            y = y.squeeze()

            # then we send the data to the gpu
            X,y = X.to(device), y.to(device)

            ## FOWARD PASS
            test_pred = model(X)

            # compute the test lossese and accumulate
            test_mae += criterion(test_pred, y).item()

    # get the loss per batch for both metrics 
    test_mae /= len(dataloader)

    return test_mae


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
         criterion:torch.nn.modules.loss,
         epochs:int,
         device:torch.device,
         writer:torch.utils.tensorboard.writer.SummaryWriter,
         scale:float,
         nbatch:int,
         normalize):
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
        "test_loss": [],
    }

    # we loop through the training and testing steps 
    for epoch in tqdm(range(epochs)):
        # run through the training step by using the function
        train_loss = train_step(model=model, 
                                dataloader=train_dataloader,
                                criterion=criterion,
                                optimizer=optimizer,
                                device=device,
                                normalize=normalize,
                                scale=scale,
                                nbatch=nbatch)
        
        # run through the corresponding testing step
        test_loss = test_step(model=model,
                              criterion=criterion,
                              dataloader=test_dataloader,
                              device=device,
                              normalize=normalize,
                              scale=scale)
        
        # print out where we are in training the model 
        print(
            f"EPOCH: {epoch + 1} | "
            f"TRAIN LOSS: {train_loss:.3f} | "
            f"TEST LOSS: {test_loss:.3f} | "
        )

        # update our dictionary
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

        # EXPIREMENT TRACKING:
        # add loss results to the SummaryWriter
        writer.add_scalars(
            main_tag="LOSS MAE",
            tag_scalar_dict={"train_mae":train_loss,
                            "test_mae":test_loss},
            global_step=epoch
        )

    writer.add_graph(
        model=model,
        input_to_model=torch.randn(32,1,820,820).to(device)
    )

    # we close the writer instance 
    writer.close()

    return results
