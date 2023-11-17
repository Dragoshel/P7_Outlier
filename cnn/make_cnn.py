import random
import torch
import os
import argparse

from cnn.create_cnn import create_cnn
from cnn.test import test_model
from dataloaders.test_loader import testing_data_loader
from dataloaders.train_loader import training_data_loaders
from utils.classes import pick_classes, get_normal_classes

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
    elif args.test:
        print('[INFO] Testing model')
        test()
    else:
        print('[WARN] CNN Model is already created, try testing it instead')
