import random
import torch
import torchvision

from utils.normalizer import get_transform
from torch.utils.data import DataLoader, Subset, ConcatDataset

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
    number_set = torchvision.datasets.MNIST(data_path + '/testing/MNIST', train=False, transform=get_transform(), download=True)
    clothes_set = torchvision.datasets.FashionMNIST(data_path + '/testing/FMNIST', train=True, transform=get_transform(), download=True)
    
    normal_set = [{'normal':features} for (features, label) in number_set if label in normal_classes]
    novelty_set = [{'novelty':features} for (features, label) in number_set if label in novel_classes]

    outlier_indices = random.sample(range(len(clothes_set)), no_outliers)
    outlier_set = Subset(clothes_set, outlier_indices)
    outlier_set = [{'outlier':features} for (features, label) in outlier_set]
    
    testing_set = ConcatDataset([normal_set, novelty_set, outlier_set])
    
    return DataLoader(testing_set, batch_size = batch_size, shuffle = True, generator=generator)