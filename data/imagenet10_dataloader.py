import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import config

# Get loder of data
def get_data_loaders():
    print('==> Preparing Imagenet 10 class data..')
    # Define the paths to the training and validation datasets
    traindir = config.imagenet10_traindir  # Path to the training data
    valdir = config.imagenet10_valdir     # Path to the validation data

    # Normalization transform: scales the image tensors to a standard range
    # These mean and std values are commonly used for ImageNet datasets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization values
    
    # Define the data loader for the training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            # Data augmentation: random resizing and cropping to 224x224
            transforms.RandomResizedCrop(224),
            # Data augmentation: random horizontal flip
            transforms.RandomHorizontalFlip(),
            # Convert images to PyTorch tensors
            transforms.ToTensor(),
            # Normalize the images using the predefined normalization
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=True,  # Shuffle data for better training
        num_workers=4, pin_memory=True)  # Use 4 worker threads and pin memory for faster data transfer

    # Define the data loader for the validation dataset
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            # Resize images to 256x256
            transforms.Resize(256),
            # Crop the center 224x224 region
            transforms.CenterCrop(224),
            # Convert images to PyTorch tensors
            transforms.ToTensor(),
            # Normalize the images using the predefined normalization
            normalize,
        ])),
        batch_size=config.batch_size, shuffle=False,  # Do not shuffle validation data
        num_workers=4, pin_memory=True)  # Use 4 worker threads and pin memory for faster data transfer

    return train_loader, val_loader
