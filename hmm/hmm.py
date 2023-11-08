from pomegranate.distributions import Normal, Categorical
from pomegranate.hmm import DenseHMM, SparseHMM

from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST, MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import torch
import numpy

N_HIDDEN_STATES = 3
N_DIMENSIONS = 28 * 28

TRAIN_SPLIT = 0.20
VALID_SPLIT = 1 - TRAIN_SPLIT

print("[INFO] Loading the MNIST Dataset...")
train_data = MNIST(root='data', train=True,
                   download=True, transform=ToTensor())
num_train_samples = int(len(train_data) * TRAIN_SPLIT)
num_valid_samples = int(len(train_data) * VALID_SPLIT)
(train_data, valid_data) = random_split(train_data,
                                        [num_train_samples, num_valid_samples],
                                        generator=torch.Generator().manual_seed(42))

print("[INFO] Initializing model...")
train_data_loader = DataLoader(train_data, shuffle=True, batch_size=num_train_samples)

# Create hmm with N_HIDDEN_STATES *empty* distributions
# which map 1 to 1 to N_HIIDEN_STATES output nodes
distributions = [Normal() for _ in range(N_HIDDEN_STATES)]
model = DenseHMM(distributions, verbose=True)

# Initialize the transition matrix with random values
model.fit(torch.randn(1, N_DIMENSIONS, 1))

for train_images, train_labels in train_data_loader:
    train_images = train_images.reshape(-1, N_DIMENSIONS, 1)

    print("[INFO] Fitting model...")
    model.fit(train_images)
