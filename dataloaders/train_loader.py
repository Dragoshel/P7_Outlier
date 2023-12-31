import random
import numpy
import torch
import torchvision
from utils.data_types import DataType

from utils.normalizer import get_transform
from torch.utils.data import Subset, DataLoader
from utils.classes import get_normal_classes

generator = torch.Generator()
generator.manual_seed(0)

def training_data_loaders(batch_size: int, data_path: str) -> tuple:
    """Downloads training and validation sets for training the models.
    The training set is shuffled to minimize the risk of overfitting the model
    to a specific image. The validation set is not shuffled to allow for proper
    validation of the model in terms of overfitting and accuracy.

    Args:
        batch_size (int): Size of the batches for each epoch
        data_path (str): Path for saving the downloaded data

    Returns:
        tuple: Training and validation dataloader
    """
    
    training_set = torchvision.datasets.MNIST(data_path + '/training', train=True, transform=get_transform(), download=True)
    validation_set = torchvision.datasets.MNIST(data_path + '/validation', train=False, transform=get_transform(), download=True)
    labels = get_normal_classes()
    training_set = Subset(training_set, [i for i, target in enumerate(training_set.targets) if target in labels])
    validation_set = Subset(validation_set, [i for i, target in enumerate(validation_set.targets) if target in labels])
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, generator=generator)

    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    return training_loader, validation_loader


