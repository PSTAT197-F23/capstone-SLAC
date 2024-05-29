'''Script for generating model predictions on real test data. Stores model predictions as a CSV file in the results folder.

Usage in terminal: 

> python test.py arg1 arg2 arg3 arg4

- arg1: number of activations in model architecture
- arg2: number of layers in ResNet
- arg3: model path (path to file containing trained model parameters)
- arg4: test data path (path to folder containing testing data; not master file)
'''

print('Importing modules...')
import sys
from resnet_arch import create_resnet
import torch
import h5py
import os
import numpy as np
import pandas as pd
from torch import nn

# store number of activations
try:
    activations = int(sys.argv[1])
except:
    print('First argument must be an integer (number of activations)')
    sys.exit(1) # halt execution
    
print('Number of activations:', activations)

# store number of layers
try:
    layers = int(sys.argv[2])
except:
    print('Second argument must be an integer (number of layers)')
    sys.exit(1) # halt execution
    
print('Number of layers:', layers)

# store command line path argument
path = sys.argv[3] 
print('Model path:', path)

# store test data path argument
test_data = sys.argv[4]

# initialize resnet-layer model
print('Initializing model...')
model = create_resnet('cpu', layers, int(activations), 0)

# load model state dict
print('Loading model state dict...')
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

# set model to evaluation mode
model.eval()

# move model to cpu device
model = model.to('cpu')

# initialize list of results
results = []

# instantiate criterion
criterion = nn.L1Loss()

# initialize metric
loss = 0

# loop through each file in test_data
for file in os.listdir(test_data):
    if file[-2:] == 'h5':

        # read in file
        h = h5py.File(os.path.join(test_data, file), 'r')

        # get number of images in file
        N = h['images'].shape[0]

        # try to get labels
        try:
            idx1 = list(h['labels'].attrs['names']).index('pitch_deg')
            idx2 = list(h['labels'].attrs['names']).index('yaw_deg')
            label_array = h['labels'][:, [idx1, idx2]]
            labels = True
            
        except:
            labels = False

        # loop through each image in the file
        for i_img in range(N):

            # store image and convert to tensor
            img = np.array(h['images'][i_img]).astype(np.float32)[None,None]
            img_t = torch.tensor(img)
            
            # make predictions
            with torch.no_grad():
                pred = model(img_t)[0]

            # store x,y tensor as two floats
            x_pred = float(pred[0])
            y_pred = float(pred[1])

            # if we have labels, print and store them in data frame
            if labels:
                label = label_array[i_img]
                x_actual, y_actual = label
                print(f'PREDICTION: ({x_pred:>9f}, {y_pred:>9f})  |  ACTUAL: ({x_actual:>9f}, {y_actual:>9f})')
    
                # compute the test losses and accumulate
                loss += criterion(pred, torch.tensor(label)).item()
                
                # store result: [file name, image index, x prediction, y prediction, x actual, y actual]
                result = [file, i_img, x_pred, y_pred, x_actual, y_actual]
                results.append(result)

            # if there are no labels
            else:
                print(f'PREDICTION: ({x_pred:>9f}, {y_pred:>9f})')
                
                # store result: [file name, image index, x prediction, y prediction]
                result = [file, i_img, x_pred, y_pred]
                results.append(result)

# get the average loss
loss /= len(results)
print(loss)

# name of results file path
results_path = f'results/preds_{path.split("/")[-1]}_{loss:.4f}.csv'

# convert results to data frame
if labels:
    results_df = pd.DataFrame(results, columns=['File Name', 'Image Index', 'X Prediction', 'Y Prediction', 'X Actual', 'Y Actual'])
else:
    results_df = pd.DataFrame(results, columns=['File Name', 'Image Index', 'X Prediction', 'Y Prediction'])

# save results data frame to CSV file
results_df.to_csv(results_path, index=False)
