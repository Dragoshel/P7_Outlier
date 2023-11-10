from pomegranate.distributions import Normal, Categorical
from pomegranate.hmm import DenseHMM, SparseHMM

from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST, MNIST
from torch.utils.data import random_split, DataLoader, Subset

import torch.nn.functional as F

from sklearn.metrics import classification_report

import torch
import numpy

N_HIDDEN_STATES = 3
N_DIMENSIONS = 28 * 28
# % of test data to test on
N_TEST_DATA = 0.05
# num of training iternations hmm will do for every batch
N_FIT_ITER = 5
# num of digit labels to train on
N_TRAIN_LABELS = 9

print("[INFO] Loading the MNIST Dataset...")
train_data = MNIST(root='data', train=True,
                   download=True, transform=ToTensor())
test_data = MNIST(root='data', train=False,
                  download=True, transform=ToTensor())

models = []

for digit in range(N_TRAIN_LABELS):
    print(f"[INFO] Initializing model for digit {digit} ...")
    train_data_subset = [img for img, label in zip(
        train_data.data, train_data.targets) if label == digit]
    train_data_loader = DataLoader(
        train_data_subset, shuffle=True, batch_size=len(train_data))

    # Create hmm with N_HIDDEN_STATES *empty* distributions
    # which map 1 to 1 to N_HIIDEN_STATES output nodes
    distributions = [Normal() for _ in range(N_HIDDEN_STATES)]
    model = DenseHMM(distributions, max_iter=N_FIT_ITER, verbose=True)

    # Let the fit method initialize the transition matrix
    # with random values
    model.fit(torch.randn(1, N_DIMENSIONS, 1))

    for train_images in train_data_loader:
        train_images = train_images.reshape(-1, N_DIMENSIONS, 1) / 255

        print(f"[INFO] Fitting model for digit {digit} ...")
        model.fit(train_images)

    models.append(model)


test_data_subset, _ = random_split(test_data,
    [int(len(test_data) * N_TEST_DATA), int(len(test_data) * (1 - N_TEST_DATA))],
    generator=torch.Generator().manual_seed(42)
)
test_data_loader = DataLoader(
    test_data_subset, shuffle=True, batch_size=len(test_data))

print(f"[INFO] Testing model with {len(test_data_subset)} datapoints ...")
y_true = numpy.array([test_data_subset.dataset.targets[i] for i in test_data_subset.indices])
y_pred = []
for test_images, _ in test_data_loader:
    for test_image in test_images:
        test_image = test_image.reshape(-1, N_DIMENSIONS, 1)
        log_probs = numpy.array([model.log_probability(test_image) for model in models])
        pred = log_probs.argmax()
        y_pred.append(pred)

print(classification_report(
    y_true=y_true,
    y_pred=y_pred,
    target_names=test_data_subset.dataset.classes)
)
