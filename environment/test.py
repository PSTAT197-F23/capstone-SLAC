'''Script for generating model predictions on real test data. Stores model predictions as a CSV file in the results folder.

Usage in terminal: 

> python test.py arg1 arg2

- arg1: number of activations in model architecture
- arg2: model path (path to file containing trained model parameters)
'''

print('Importing modules...')
import sys
from resnet_arch import create_resnet
import torch
import h5py
import os
import numpy as np
import pandas as pd

# store number of activations
try:
    activations = int(sys.argv[1])
except:
    print('First argument must be an integer (number of activations)')
    sys.exit(1) # halt execution
    
print('Number of activations:', activations)

# store command line path argument
path = sys.argv[2] 
print('Model path:', path)

# initialize resnet18 model
print('Initializing model...')
model = create_resnet('cpu', 18, int(activations), 0)

# load model state dict
print('Loading model state dict...')
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

# set model to evaluation mode
model.eval()

# move model to cpu device
model = model.to('cpu')

# initialize list of results
results = []

# loop through each file in test_data
for file in os.listdir('test_data'):
    if file[:4] == 'rank':

        # read in file
        h = h5py.File('test_data/'+file, 'r')

        # get number of images in file
        N = h['images'].shape[0]

        # loop through each image in the file
        for i_img in range(N):

            # store image and convert to tensor
            img = np.array(h['images'][i_img]).astype(np.float32)[None,None]
            img_t = torch.tensor(img)
            
            # make predictions
            with torch.no_grad():
                pred = model(img_t)[0]

            # store x,y tensor as two floats
            x = float(pred[0])
            y = float(pred[1])
            print(x, y)

            # store result: [file name, image index, (x prediction, y prediction)]
            result = [file, i_img, (x, y)]
            results.append(result)

# name of results file path
results_path = f'results/preds_{path.split("/")[-1]}.csv'

# convert results to data frame
results_df = pd.DataFrame(results, columns=['File Name', 'Image Index', 'Prediction'])

# save results data frame to CSV file
results_df.to_csv(results_path, index=False)
