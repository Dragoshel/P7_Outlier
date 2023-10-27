import copy
import random
import torch
import torchvision
from dataloaders.test_dataset import TestDataset
from utils.data_types import DataType

from utils.normalizer import get_transform
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch import clone

generator = torch.Generator()
generator.manual_seed(0)

def testing_data_loader(batch_size: int, data_path: str, no_outliers: int, normal_classes: list, novel_classes: list) -> DataLoader:
    """ Downloads the testing data sets and saves it to the specified folder

    Args:
        batch_size (int): Size of the batches for each epoch
        data_path (str): Path for saving the downloaded data

    Returns:
        DataLoader: Testing dataloader
    """
    dataset = TestDataset(normal_classes, novel_classes, no_outliers, data_path)
    
    print('Testing set has {} instances'.format(len(dataset)))
    
    return DataLoader(dataset, batch_size = batch_size, shuffle = True, generator=generator)