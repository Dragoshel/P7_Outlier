import copy
import random
import torch
from torch.utils.data import Dataset

import torchvision
from torch.utils.data import Dataset
from utils.data_types import DataType
from utils.normalizer import get_transform

class TestDataset(Dataset):
    def __init__(self, normal_classes: list, novel_classes: list, no_outliers: list, data_path: str):
        number_set = torchvision.datasets.MNIST(data_path + '/testing/MNIST', train=False, transform=get_transform(), download=True)
        normal_images, normal_labels = self._filter_dataset(copy.deepcopy(number_set), normal_classes, DataType.NORMAL)
        novel_images, novel_labels = self._filter_dataset(copy.deepcopy(number_set), novel_classes, DataType.NOVEL)
        
        clothes_set = torchvision.datasets.FashionMNIST(data_path + '/testing/FMNIST', train=True, transform=get_transform(), download=True)
        outlier_images, outlier_labels = self._filter_dataset(clothes_set, range(10), DataType.OUTLIER)
        outlier_images, outlier_labels = zip(random.sample(list(zip(outlier_images, outlier_labels)), no_outliers))
        
        self.images = torch.cat(normal_images, novel_images, outlier_images)
        self.labels = torch.cat(normal_labels, novel_labels, outlier_labels)
    
    def _filter_dataset(self, data_set: Dataset, classes: list, datatype: DataType) -> Dataset:
        idx = [label for label in data_set.targets if label in classes]
        data_set.data = data_set.data[idx]
        data_set.targets[idx] = datatype
        return data_set.data, data_set.targets
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index) -> tuple:
        images = self.images[index]
        labels = self.labels[index]
        return images, labels
    