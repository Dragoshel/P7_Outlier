import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST

from cnn.create_cnn import create_cnn
from cnn.test import test_model
from dataloaders.test_loader import testing_data_loader
from dataloaders.train_loader import training_data_loaders
from utils.classes import get_normal_classes, pick_classes
from utils.normalizer import get_transform
from torch.utils.data import Subset, DataLoader

# Counteract non-determinism
random.seed(10)
torch.manual_seed(10)

# Pick normal, abnormal and novelty class labels
no_norms = 8
pct_outliers = 0.2
no_outliers = int(50000 / (1-pct_outliers) * pct_outliers)

# Set train and size and run throughs
batch_size = 64
num_epochs = 10
data_path = './data'

model_path = 'models/cnn_model.pth'

def train():
    print('[INFO] Creating train and validation sets')
    training_loader, validation_loader = training_data_loaders(batch_size, data_path)
    
    cnn_model = create_cnn(num_epochs, training_loader, validation_loader, no_norms)
    print('[INFO] Finished training, saving model...')
    torch.save(cnn_model, model_path)

def test():
    cnn_model = torch.load(model_path)
    test_loader, labels = testing_data_loader(1, data_path, no_outliers)
    test_model(test_loader, cnn_model, labels)

def threshold(classes, test_data_size=5000, density=10):
    generator = torch.Generator()
    generator.manual_seed(10)

    # test_data = FashionMNIST(root='data', train=True, download=True, transform=get_transform())
    test_data = MNIST(root='data', train=True, download=True, transform=get_transform())
    test_data = Subset(test_data, [i for i, target in enumerate(test_data.targets) if target in classes])

    (test_data, _) = random_split(test_data,
        [test_data_size, len(test_data) - test_data_size], generator=generator)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, generator=generator)

    model = torch.load(model_path)
    model.eval()

    print(f'[INFO] Generating threshold on {test_data_size} datapoints...')
    print(f'[INFO] Density of the thresholds is {density} ...')
    thresholds = [0] * density
    for images, targets in test_loader:
        pred = model(images)
        probs = F.softmax(pred, dim=1)
        for prob in probs:
            max_prob = torch.max(prob)
            max_prob = int(max_prob * density)
            thresholds[max_prob] += 1
    for threshold in thresholds:
        print('{:.2f}'.format(threshold / test_data_size), end=', ')

def parse():
    parser = argparse.ArgumentParser(
        prog='CNN classifier',
        description='Builds a CNN for the randomly chosen classes',
        epilog='No worries...'
    )

    parser.add_argument('-t', '--test', action='store_true', help='Flag for testing the saved CNN model')
    parser.add_argument('-m', '--make', action='store_true', help='Flag for creating the CNN model')

if __name__ == '__main__':
    args = parse()

    print('[INFO] Picking classes')
    pick_classes(no_norms)

    if not os.path.exists(model_path) and args.make:
        print('[INFO] Training model')
        train()
    elif not os.path.exists(model_path) and args.test:
        print('[INFO] Training model')
        train()
        print('[INFO] Testing model')
        test()
    elif True:
        print('[INFO] Testing model')
        test()
    else:
        print('[WARN] CNN Model is already created, try testing it instead')
