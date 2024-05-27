# we will write a helper function to save a model. 
# This will be extremely important when we expirement track 
import torch 
from pathlib import Path

def save_model(model:torch.nn.Module, target_dir:str, model_name:str):
    '''
        Saves a PyTorch model to a specified directory. The directory will be models 

        Parameters:
            model: The model that we want to save.
            target_dir: The directory where we want to save the model (A folder)
            model_name: The filename we want for the model. (Include the extension.)

        Example Usage: 
            save_model(model = baseline_model, 
                       target_dir = "models", 
                       model_name = "baseline_model.pth")
    '''
    # we first create a target directory path 
    target_dir_path = Path(target_dir)

    # we then create the directory
    # the parents = True ensures that the directory will be created if one does not exists
    # the exist_ok = True ensures that if the directory does exist then no error occurs
    target_dir_path.mkdir(parents=True,
                         exist_ok=True)

    # we then make the model saved path but we first check that the user did this correctly
    try:
        assert model_name.endswith(".pth") or model_name.endswith(".pt")
    except AssertionError:
        print('THE MODEL_NAME MUST END WITH .pth or pt')

    # we then create the path
    saved_path = target_dir_path / model_name

    # we finally save the model state_dict()
    print(f'[INFO] Saving model to {saved_path}')
    torch.save(obj=model.state_dict(),
              f = saved_path)
    
