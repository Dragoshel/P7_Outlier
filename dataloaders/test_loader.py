import math

import torch
import torchvision

from learn2learn.data import MetaDataset, UnionMetaDataset
from torch.utils.data import DataLoader, Dataset, random_split

from torchvision.transforms import PILToTensor



def testing_data_loader(no_outliers: int) -> (DataLoader, list):
    generator = torch.Generator()
    generator.manual_seed(10)
    torch.manual_seed(10)
    dataset, labels = _union_set(no_outliers)
    batch_size = int(math.ceil(len(dataset)/20))
    
    print('Testing set has {} instances'.format(len(dataset)))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator), labels

def _union_set(no_outliers: int) -> (Dataset, list):  
    generator = torch.Generator()
    generator.manual_seed(10)
    torch.manual_seed(10)
    clothes_set = torchvision.datasets.FashionMNIST('testing/FMNIST', train=True, download=True, transform=PILToTensor())
    discard = len(clothes_set) - no_outliers
    outlier_set, _ = random_split(clothes_set,
        [no_outliers, discard],
        generator=generator
    )
    outlier_set = MetaDataset(outlier_set)

    number_set = torchvision.datasets.MNIST('testing/MNIST', train=False, download=True, transform=PILToTensor())
    number_set = MetaDataset(number_set)

    union = UnionMetaDataset([number_set, outlier_set])

    return union, number_set.labels
