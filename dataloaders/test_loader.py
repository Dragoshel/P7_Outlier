import random

import torch
import torchvision

from learn2learn.data import MetaDataset, UnionMetaDataset
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

from utils.normalizer import get_transform

generator = torch.Generator()
generator.manual_seed(0)

def testing_data_loader(batch_size: int, data_path: str, no_outliers: int) -> (DataLoader, list):
    """ Downloads the testing data sets and saves it to the specified folder

    Args:
        batch_size (int): Size of the batches for each epoch
        data_path (str): Path for saving the downloaded data

    Returns:
        DataLoader: Testing dataloader
    """
    dataset, labels = _union_set(no_outliers, data_path)
    
    print('Testing set has {} instances'.format(len(dataset)))
    
    return DataLoader(dataset, batch_size = batch_size, shuffle = True, generator=generator), labels

def _union_set(no_outliers: int, data_path: str) -> (Dataset, list):
    clothes_set = torchvision.datasets.FashionMNIST(data_path + '/testing/FMNIST', train=True, transform=get_transform(), download=True)
    chosen_outliers = random.sample(range(len(clothes_set)), no_outliers)
    outlier_set = Subset(clothes_set, chosen_outliers)
    outlier_set = MetaDataset(outlier_set)

    number_set = torchvision.datasets.MNIST(data_path + '/testing/MNIST', train=False, transform=get_transform(), download=True)
    number_set = MetaDataset(number_set)

    # the union order doesnt mach the dataset order
    # union = UnionMetaDataset([number_set, outlier_set])
    
    union = UnionMetaDataset([number_set, outlier_set])



    # union = ConcatDataset([number_set, outlier_set])
    # print(outlier_set.targets)
    return union, number_set.labels
    # return number_set works fine