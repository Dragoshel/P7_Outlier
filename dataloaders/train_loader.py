import torch
import torchvision

from torchvision.transforms import PILToTensor
from torch.utils.data import Subset, DataLoader
from utils.classes import get_normal_classes

generator = torch.Generator()
generator.manual_seed(0)

def training_data_loaders(batch_size: int, data_path: str) -> tuple:
    training_set = torchvision.datasets.MNIST(data_path + '/training', train=True, transform=PILToTensor(), download=True)
    validation_set = torchvision.datasets.MNIST(data_path + '/validation', train=False, transform=PILToTensor(), download=True)
    labels = get_normal_classes()
    training_set = Subset(training_set, [i for i, target in enumerate(training_set.targets) if target in labels])
    validation_set = Subset(validation_set, [i for i, target in enumerate(validation_set.targets) if target in labels])
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, generator=generator)

    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    return training_loader, validation_loader


