from pomegranate.distributions import Normal, Categorical
from pomegranate.hmm import DenseHMM, SparseHMM

from torchvision.transforms import transforms, ToTensor
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

random.seed(10)
torch.manual_seed(10)
generator = torch.Generator()
generator.manual_seed(10)

_norm_factor = 0.5
# Transformations describing how we wish the data set to look
# Normalize is chosen to normalize the distribution of the image tiles
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((_norm_factor,), (_norm_factor,))
])

def get_norm_factor() -> float:
    return _norm_factor

# Image dimensions for flattening
N_DIMENSIONS = 28 * 28

# Grid size
GRID_SIZE = 4

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"[INFO] Running on {device.type}...")

print("[INFO] Loading the MNIST Dataset ...")
test_data = MNIST(root='testing', train=False,
                  download=True, transform=_transform)

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

def priors_hmm(models, images):
    global device
    print(f"Getting priors from HMM")
    test_images = test_images.reshape(-1, N_DIMENSIONS, 1)
    test_images = test_images.to(torch.int64)
    test_images = reform_images(test_images)

    probs = numpy.array([model.log_probability(test_images).tolist() for model in models])
    probs = probs.transpose()
    return probs

def load_models(folder_path: str) -> list:
    models = []
    for digit in range(args.num_classes):
        model = torch.load(f'{folder_path}/model{digit}.pth').to(device)
        models.append(model)
    return models
