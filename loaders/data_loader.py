import torch
import torchvision
import torchvision.transforms as transforms

_norm_factor = 0.5
# Transformations describing how we wish the data set to look
# Normalize is chosen to normalize the distribution of the image tiles
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((_norm_factor,), (_norm_factor,))
])

def get_norm_factor() -> float:
    return _norm_factor

def training_data_loaders(batch_size: int, data_path: str) -> tuple:
    """Downloads training and validation sets for training the models.
    The training set contains 60k images and is shuffled to minimize the risk of overfitting the model
    to a specific image. The validation set contains 10k images, and is not shuffled to allow for proper
    validation of the model in terms of overfitting and accuracy.

    Args:
        batch_size (int): Size of the batches for each epoch
        data_path (str): Path for saving the downloaded data

    Returns:
        tuple: Training and validation dataloader
    """    
    training_set = torchvision.datasets.MNIST(data_path + '/training', train=True, transform=_transform, download=True)
    validation_set = torchvision.datasets.MNIST(data_path + '/validation', train=False, transform=_transform, download=True)
    
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    
    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    return training_loader, validation_loader

def testing_data_loader(batch_size: int, data_path: str) -> torch.utils.data.DataLoader:
    """ Downloads the testing data sets and saves it to the specified folder

    Args:
        batch_size (int): Size of the batches for each epoch
        data_path (str): Path for saving the downloaded data

    Returns:
        DataLoader: Testing dataloader
    """    
    testing_set = torchvision.datasets.MNIST(data_path + '/testing', train=False, transform=_transform, download=True)
    
    return torch.utils.data.DataLoader(dataset = testing_set, batch_size = batch_size, shuffle = True)
