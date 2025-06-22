"""
main_unet.py

Entry point for voxel-level brain age prediction pretraining using the MONAI 3D UNet architecture.

Main features:
- Sets up the environment, device, and Weights & Biases (wandb) experiment tracking.
- Initializes the 3D UNet model for brain age prediction.
- Loads training, validation, and test datasets and dataloaders.
- Configures optimizer and learning rate scheduler.
- Supports resuming training from the last saved checkpoint.
- Runs the full training loop and evaluates the model on the test set.

Functions:
    main():
        Initializes all components, runs training, and performs final testing.
    """


import os
import torch
import wandb
from monai.config import print_config
from monai.networks.nets import UNet

from transforms import *
from load_data import *
from loss import *
from train import *
from testfunction import *
from testfunction import test


print_config()
# Set the PYTORCH_CUDA_ALLOC_CONF to avoid memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# Set CUDA launch blocking to help with debugging
torch.backends.cudnn.benchmark = True
#CUDA_LAUNCH_BLOCKING = 1
os.environ['TORCH_USE_CUDA_DSA'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import (
    UNET_OUTPUT_DIR, UNET_PROJECT_NAME,
    FULL_DATA_CSV, TEST_CSV,
    UNET_OPTIMIZER_CLASS, UNET_OPTIMIZER_PARAMS,
    UNET_SCHEDULER_CLASS, UNET_SCHEDULER_PARAMS,
    UNET_MAX_EPOCHS
)
# Set up Weights & Biases
if wandb.run is None:
    

    wandb.init(project=UNET_PROJECT_NAME, settings=wandb.Settings(start_method="fork", _service_wait=800))



    wandb.run.name = 'x'


def main():
    """
    Main function to initialize the model, load data, and run training and testing phases.

    Returns:
        None
    """
    # Specify the name of the directory to save models and logs
    directory_name = UNET_OUTPUT_DIR

    # Create the directory
    os.makedirs(directory_name, exist_ok=True)

    # Initialize your model, optimizer, scheduler, etc.


    # Initialize the UNet model for 3D input of size (128, 160, 128)
    model = UNet(
        spatial_dims=3,  # 3D input for dimensions (128, 160, 128)
        in_channels=1,   # Number of input channels
        out_channels=1,  # Number of output channels
        channels=(16, 32, 64, 128, 256),  # Channel sizes; adjust these as needed
        strides=(2, 2, 2, 2),  # Strides for downsampling
        num_res_units=2,  # Number of residual units per layer
    )
       # Use DataParallel if multiple GPUs are available
    model = torch.nn.DataParallel(model).to(device)
    
   # Initialize optimizer and scheduler
  
    optimizer = UNET_OPTIMIZER_CLASS(model.parameters(), **UNET_OPTIMIZER_PARAMS)
    scheduler = UNET_SCHEDULER_CLASS(optimizer, **UNET_SCHEDULER_PARAMS)
    max_epochs = UNET_MAX_EPOCHS

    # Load the last model weights (if available)
    model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss = load_last_model(model, optimizer, scheduler, directory_name)

   # Call the load_data function to create datasets and dataloaders
    ds_train, train_loader, ds_val, val_loader, ds_test, test_loader = load_data(
        full_data_path=FULL_DATA_CSV,
        test_set_path=TEST_CSV
    )
    # Start training
    train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=start_epoch)

    # Test the trained model
    test(test_loader, model, directory_name, TEST_CSV)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    main()


