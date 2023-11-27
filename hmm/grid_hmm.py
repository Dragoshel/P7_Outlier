from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM

from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST, MNIST, FashionMNIST
from torch.utils.data import random_split, DataLoader, Subset

import torch.nn.functional as F

from sklearn.metrics import classification_report

import argparse
import torch
import numpy
import random
import os

random.seed(10)
torch.manual_seed(10)
generator = torch.Generator()
generator.manual_seed(10)

# Image sizes for reformating
GRID_SIZE = 4
IMG_SIZE = 28

# Model distribution initialisation parameters
N_OBSERVATIONS = 50
PREFERRED_SUM = 0.8
N_DISTRIBUTIONS = 50
N_DIMENSIONS = IMG_SIZE **2

# train data % subset
N_TRAIN_DATA = 0.1
TRAIN_AMOUNT = int(50000*N_TRAIN_DATA)

print("[INFO] Loading the MNIST Dataset ...")
train_data = MNIST(root='data/hmm/training', train=True, download=True, transform=ToTensor())
test_data = MNIST(root='data/hmm/testing', train=False, download=True, transform=ToTensor())

def make_grid(image):
    image = torch.flatten(image)
    image = torch.reshape(image, [28,28])
    h,w = image.shape
    h =(h // GRID_SIZE) * GRID_SIZE
    w = (w // GRID_SIZE) * GRID_SIZE
    image = image[:h, :w]
    image = image.reshape(h // GRID_SIZE, GRID_SIZE, -1, GRID_SIZE).swapaxes(1, 2).reshape(h // GRID_SIZE, w // GRID_SIZE, -1).sum(axis=-1) % N_OBSERVATIONS
    return image

def reform_images(images: list):
    new_images = []
    grid_sq = 28 // GRID_SIZE
    for image in images:
        new_image = make_grid(image)
        new_image = new_image.reshape(grid_sq**2, 1)
        new_images.append(new_image.tolist())
    return torch.tensor(new_images)

def create_distributions(num_dist, pref_sum, count):
    uniform_dist = Categorical([(numpy.ones(count) / count)])
    dists = [uniform_dist]
    for i in range(num_dist):
        pref_size = int(count / 2 ** i)
        unpref_size = count - pref_size
        pref_part = numpy.array([pref_sum] * pref_size) / pref_size
        unpref_part = numpy.array([1 - pref_sum] * unpref_size) / unpref_size
        dist = Categorical([numpy.concatenate([unpref_part, pref_part])])
        dists.append(dist)
    return dists

def train_model(digit):
    model_path = f'models/model{digit}.pth'

    if not os.path.exists(model_path):
        print(f"[INFO] Initializing model for digit {digit} ...")
        # Create train set for digit with predefined amount
        train_data_subset = [img for img, label in zip(
            train_data.data, train_data.targets) if label == digit]
        
        keep = int(len(train_data_subset) * N_TRAIN_DATA)
        discard = int(len(train_data_subset) - keep)
        train_data_subset, _ = random_split(train_data_subset,
            [keep, discard],
                generator=generator
        )

        train_data_loader = DataLoader(
            train_data_subset, shuffle=True, batch_size=TRAIN_AMOUNT, generator=generator)

        # Setting up the base model
        distributions = create_distributions(
            N_DISTRIBUTIONS, PREFERRED_SUM, N_OBSERVATIONS)
        model = DenseHMM(distributions, verbose=True)

        # Train model to fit sequences observed in a single number
        print(f"[INFO] Fitting model for digit {digit} ...")
        for train_images in train_data_loader:
            # Reshape images to match (batch_size, 784, 1) with int values
            train_images = train_images.reshape(-1, N_DIMENSIONS, 1)
            train_images = train_images.to(torch.int64)
            # Format images to grids
            train_images = reform_images(train_images)

            model.fit(train_images)
        torch.save(model, model_path)

def test(models):
    test_data_loader = DataLoader(test_data.data,
        shuffle=False, batch_size=1000, generator=generator)
    
    print(f"[INFO] Testing model with {len(test_data)} datapoints ...")
    y_true = test_data.targets
    y_pred = []
    for i, test_images in enumerate(test_data_loader):
        # Reshape images to match (batch_size, 784, 1) with int values
        test_images = test_images.reshape(-1, N_DIMENSIONS, 1)
        test_images = test_images.to(torch.int64)
        # Format images to grids
        test_images = reform_images(test_images)

        probs = numpy.array([model.log_probability(test_images).tolist() for model in models])
        probs = probs.transpose()
        preds = [prob.argmax() for prob in probs]
        y_pred.extend(preds)

    print(classification_report(
       y_true=y_true,
       y_pred=y_pred,
       target_names=test_data.classes)
    )

def threshold(models, classes, test_data_size=5000, density=10):
    # test_data = MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_data = MNIST(root='data', train=False, download=True, transform=ToTensor())
    # test_data = KMNIST(root='data', train=True, download=True, transform=ToTensor())
    # test_data = FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
    test_data = Subset(test_data.data, [i for i, target in enumerate(test_data.targets) if target in classes])

    # (test_data, _) = random_split(test_data,
    #     [test_data_size, len(test_data) - test_data_size], generator=generator)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=True, generator=generator)
    thresholds = [0] * (density + 1)

    for i, test_images in enumerate(test_loader):
        test_images = test_images.reshape(-1, N_DIMENSIONS, 1)
        test_images = test_images.to(torch.int64)

        test_images = reform_images(test_images)

        probs = numpy.array([model.log_probability(test_images).tolist() for model in models])
        probs = torch.tensor(probs.transpose())
        probs = F.softmax(probs, dim=1)

        for prob in probs:
            max_prob = torch.max(prob)
            max_prob = int(max_prob * density)
            thresholds[max_prob] += 1

    for threshold in thresholds:
        print('{:.2f}'.format(threshold / test_data_size), end=', ')


def parse():
    parser = argparse.ArgumentParser(
        prog='HMM classifier',
        description="Builds a number of HMM's and creates a probability distribution over MNIST digits",
        epilog='Hi mom!'
    )

    parser.add_argument('-c', '--num-classes', type=int, help='the number of output classes', default=10)
    parser.add_argument('-d', '--digit', type=int, help='digit to train model on')
    parser.add_argument('-a', '--all', action='store_true', help='whether to train models for all digits')
    parser.add_argument('-t', '--test', action='store_true', help='testing the entire model')

    return parser.parse_args()

def main():
    args = parse()

    if args.test:
        models = []
        for digit in range(args.num_classes):
            model = torch.load(f'output/model{digit}.pth')
            models.append(model)

        test(models)
        return

    if args.all:
        for digit in range(args.num_classes):
            train_model(digit)
    else:
        train_model(args.digit)

if __name__ == '__main__':
    main()
