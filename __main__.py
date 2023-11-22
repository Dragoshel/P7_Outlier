import argparse
import random

import torch
import torch.nn.functional as F
from cnn.create_cnn import create_cnn
from cnn.test import test_model
from dataloaders.test_loader import testing_data_loader
from dataloaders.train_loader import training_data_loaders
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms
from utils.classes import pick_classes, get_normal_classes, get_novel_classes

from hmm import grid_hmm
from cnn import make_cnn
import numpy

random.seed(10)
torch.manual_seed(10)
generator = torch.Generator()
generator.manual_seed(10)

# Pick normal, abnormal and novelty class labels
NO_NORMS = 5

def parse():
    parser = argparse.ArgumentParser(
        prog='Run different commands on P7_Outlier models.',
        description='Run different commands on P7_Outlier models.',
        epilog='Hi mom!'
    )

    subparsers = parser.add_subparsers(dest='model', help='Model type')

    parser_cnn = subparsers.add_parser('cnn', help='Run commands on a convolutional neural network.')
    parser_cnn.add_argument('--test', action='store_true', help='Test the CNN model.')
    parser_cnn.add_argument('--train', action='store_true', help='Train the CNN model.')
    parser_cnn.add_argument('--threshold', action='store_true', help='Find the CNN treshold.')

    parser_hmm = subparsers.add_parser('hmm', help='Run commands on a hidden markov model.')
    parser_hmm.add_argument('--test', action='store_true', help='Test the HMM model')
    parser_hmm.add_argument('--train-digit', action='store_true', help='Digit to train model on.')
    parser_hmm.add_argument('--train-all', action='store_true', help='Whether to train models for all digits.')
    parser_hmm.add_argument('--num-classes', type=int, help='the number of output classes', default=10)

    parser_hmm.add_argument('--threshold', action='store_true', help='Find the HMM treshold.')

    return parser.parse_args()


def main():
    args = parse()

    print('[INFO] Picking normal classes...')
    pick_classes(NO_NORMS)
    normal_classes = sorted(get_normal_classes())
    novelty_classes = sorted(get_novel_classes())

    if args.model == 'cnn':
        if args.train:
            print('[INFO] Training CNN...')
            make_cnn.train()
        elif args.test:
            print('[INFO] Testing CNN...')
            make_cnn.test()
        elif args.threshold:
            # make_cnn.threshold([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], density=10)
            make_cnn.threshold(normal_classes, density=10)
            # make_cnn.threshold(novelty_classes, density=10)

    elif args.model == 'hmm':
        models = []
        for digit in normal_classes:
            model = torch.load(f'models/model{digit}.pth')
            models.append(model)

        if args.train_all:
            for digit in normal_classes:
                grid_hmm.train_model(digit)

        elif args.train_digit:
            grid_hmm.train_model(args.train_digit)

        elif args.test:
            grid_hmm.test(models)

        elif args.threshold:
            # grid_hmm.threshold(models, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], density=10)
            # grid_hmm.threshold(models, novelty_classes, density=10)
            grid_hmm.threshold(models, normal_classes, density=10)

if __name__ == "__main__":
    main()
