from pomegranate.distributions import Normal, Categorical
from pomegranate.hmm import DenseHMM, SparseHMM

from torchvision.transforms import transforms, ToTensor
from torchvision.datasets import KMNIST, MNIST
from torch.utils.data import random_split, DataLoader, Subset

import torch.nn.functional as F

from sklearn.metrics import classification_report

import torch
import numpy
import random
import math
import os

_norm_factor = 0.5
# Transformations describing how we wish the data set to look
# Normalize is chosen to normalize the distribution of the image tiles
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((_norm_factor,), (_norm_factor,))
])


def get_norm_factor() -> float:
    return _norm_factor


# Model distribution initialisation parameters
N_OBSERVATIONS = 256
PREFERRED_SUM = 0.8
N_DISTRIBUTIONS = 10

N_DIMENSIONS = 28 * 28
# num of training iternations hmm will do for every batch
N_FIT_ITER = 100
# num of models
N_MODELS = 10
# train data % subset
N_TRAIN_DATA = 0.50

print("[INFO] Loading the MNIST Dataset...")
train_data = MNIST(root='training', train=True,
                   download=True, transform=_transform)
test_data = MNIST(root='testing', train=False,
                  download=True, transform=_transform)


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


models = []

for digit in range(N_MODELS):
    model_path = f'output/model{digit}.pth'

    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        print(f"[INFO] Initializing model for digit {digit} ...")
        train_data_subset = [img for img, label in zip(
            train_data.data, train_data.targets) if label == digit]
        keep = int(len(train_data_subset) * N_TRAIN_DATA)
        discard = int(len(train_data_subset) - keep)
        train_data_subset, _ = random_split(train_data_subset,
            [keep, discard],
                generator=torch.Generator().manual_seed(42)
        )

        train_data_loader = DataLoader(
            train_data_subset, shuffle=True, batch_size=len(train_data_subset))

        # Setting up the base model
        distributions = create_distributions(
            N_DISTRIBUTIONS, PREFERRED_SUM, N_OBSERVATIONS)
        model = DenseHMM(distributions, max_iter=N_FIT_ITER, verbose=True)

        # Train model to fit sequences observed in a single number
        for train_images in train_data_loader:
            print(f"[INFO] Fitting model for digit {digit} ...")

            train_images = train_images.reshape(-1, N_DIMENSIONS, 1)
            train_images = train_images.to(torch.int64)

            model.fit(train_images)

        torch.save(model, model_path)

    models.append(model)


test_data_loader = DataLoader(test_data.data, shuffle=True, batch_size=100)

print(f"[INFO] Testing model with {len(test_data)} datapoints ...")
y_true = test_data.targets
y_pred = []
for i, test_images in enumerate(test_data_loader):
    print(f"Batch {i}")
    test_images = test_images.reshape(-1, N_DIMENSIONS, 1)
    test_images = test_images.to(torch.int64)

    probs = numpy.array([model.log_probability(test_images)
                         for model in models])
    probs = probs.transpose()

    pred = [prob.argmax() for prob in probs]
    y_pred.extend(pred)

print(classification_report(
    y_true=y_true,
    y_pred=y_pred,
    target_names=test_data.classes)
)
