import random
import torch
import os
import argparse

from cnn.create_cnn import create_cnn
from cnn.test import *
from dataloaders.test_loader import testing_data_loader
from dataloaders.train_loader import training_data_loaders
from utils.classes import pick_classes, get_normal_classes
from torchvision.transforms import transforms, ToTensor
import torch.nn.functional as F

from torchvision.datasets import KMNIST, MNIST
from torch.utils.data import random_split, DataLoader, Subset

import torch.nn.functional as F

from sklearn.metrics import classification_report

import argparse
import torch
import numpy
import random
import math
import os

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
    training_loader, validation_loader = training_data_loaders(
        batch_size, data_path)

    cnn_model = create_cnn(num_epochs, training_loader,
                           validation_loader, no_norms)
    print('[INFO] Finished training, saving model...')
    torch.save(cnn_model, model_path)


def test():
    cnn_model = torch.load(model_path)
    test_loader, labels = testing_data_loader(1, data_path, no_outliers)
    test_model(test_loader, cnn_model, labels)


TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.20

norm_factor = 0.5
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((norm_factor,), (norm_factor,))
])

generator = torch.Generator()
generator.manual_seed(42)


def parse():
    parser = argparse.ArgumentParser(
        prog='CNN classifier',
        description='Builds a CNN for the randomly chosen classes',
        epilog='No worries...'
    )

    parser.add_argument('-t', '--test', action='store_true', help='Flag for testing the saved CNN model')
    parser.add_argument('-m', '--make', action='store_true', help='Flag for creating the CNN model')
    parser.add_argument('-x', '--threshold', action='store_true', help='Flag for creating the CNN model')

    return parser.parse_args()

def main():
    args = parse()

    print('[INFO] Picking normal classes...')
    pick_classes(no_norms)
    labels = get_normal_classes()

    print('[INFO] Loading datasets...')
    train_data = MNIST(root='data', train=True,
                       download=True, transform=transform) 
    test_data = MNIST(root='data', train=False,
                      download=True, transform=transform)

    num_train_samples = int(len(train_data) * TRAIN_SPLIT)
    num_test_samples = int(len(test_data) * TEST_SPLIT)

    train_data = Subset(train_data, [i for i, target in enumerate(
        train_data.targets) if target in labels])
    (train_data, valid_data) = random_split(train_data,
        [num_train_samples, len(train_data) - num_train_samples],
        generator=generator)

    (test_data, _) = random_split(test_data,
        [num_test_samples, len(test_data) - num_test_samples])

    train_loader = DataLoader(
        train_data, batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(
        valid_data, batch_size, shuffle=False, generator=generator)

    test_loader = DataLoader(
        test_data, batch_size, shuffle=False, generator=generator)

    if args.make:
        print('[INFO] Creating train and validation sets')
        model = create_cnn(num_epochs, train_loader, valid_loader, len(labels))

        print('[INFO] Finished training, saving model...')
        torch.save(model, model_path)
    elif args.test:
        print('[INFO] Testing model...')
        test()
    elif args.threshold:
        model = torch.load(model_path)
        model.eval()

        thresholds = [0] * 50
        print(len(test_data))

        with torch.no_grad():
            for images, targets in test_loader:
                pred = model(images)
                probs = F.softmax(pred, dim=1)
                for prob in probs:
                    max_prob = torch.max(prob)
                    max_prob = int(max_prob * 50)
                    thresholds[max_prob] += 1
        for threshold in thresholds:
            print('{:.2f}'.format(threshold / len(test_data)), end=', ')

if __name__ == '__main__':
    main()
