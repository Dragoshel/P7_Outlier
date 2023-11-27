import random

import torch
import torchvision

from learn2learn.data import MetaDataset, UnionMetaDataset
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision.transforms import ToTensor, PILToTensor

generator = torch.Generator()
generator.manual_seed(10)


def testing_data_loader(batch_size: int, data_path: str, no_outliers: int) -> (DataLoader, list):
    dataset, labels = _union_set(no_outliers, data_path)
    
    print('Testing set has {} instances'.format(len(dataset)))
    
    return DataLoader(dataset, batch_size = batch_size, shuffle = True, generator=generator), labels

def _union_set(no_outliers: int, data_path: str) -> (Dataset, list):
    clothes_set = torchvision.datasets.FashionMNIST(data_path + '/testing/FMNIST', train=True, download=True, transform=PILToTensor())
    chosen_outliers = random.sample(range(len(clothes_set)), no_outliers)
    outlier_set = Subset(clothes_set, chosen_outliers)
    outlier_set = MetaDataset(outlier_set)

    number_set = torchvision.datasets.MNIST(data_path + '/testing/MNIST', train=False, download=True, transform=PILToTensor())
    number_set = MetaDataset(number_set)

    union = UnionMetaDataset([number_set, outlier_set])

    return union, number_set.labels
