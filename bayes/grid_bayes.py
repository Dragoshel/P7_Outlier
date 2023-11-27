from pomegranate.distributions import Normal, Categorical
from pomegranate.bayes_classifier import BayesClassifier

from torchvision.transforms import transforms, ToTensor
from torchvision.datasets import KMNIST, MNIST
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

_norm_factor = 0.5
# Transformations describing how we wish the data set to look
# Normalize is chosen to normalize the distribution of the image tiles
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((_norm_factor,), (_norm_factor,))
])

def get_norm_factor() -> float:
    return _norm_factor

# Image sizes for reformating
GRID_SIZE = 4
IMG_SIZE = 28

# Model distribution initialisation parameters
N_OBSERVATIONS = 2
PREFERRED_SUM = 0.5
N_DISTRIBUTIONS = 4
N_DIMENSIONS = IMG_SIZE **2

# train data % subset
N_TRAIN_DATA = 0.1
TRAIN_AMOUNT = int(50000*N_TRAIN_DATA)

fake_priors = [[0.01, 0.2, 0.6, 0.19],
               [0.1, 0.1, 0.6, 0.2]]

print("[INFO] Loading the MNIST Dataset ...")
train_data = MNIST(root='data/bayes/training', train=True,
                   download=True, transform=_transform)
test_data = MNIST(root='data/bayes/testing', train=False,
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

def create_distributions(num_dist, pref_sum, count):
    dists = []
    for i in range(num_dist):
        uniform_dist = Categorical([(numpy.ones(count) / count)])
        dists.append(uniform_dist)
    return dists

def make_model():
    print(f"[INFO] Initializing bayes model...")
    # Setting up the base model
    distributions = create_distributions(
        N_DISTRIBUTIONS, PREFERRED_SUM, N_OBSERVATIONS)
    model = BayesClassifier(distributions)
    return model

def posterior_distribution(hmm_priors, cnn_priors):
    new_priors = []
    for hmm_val, cnn_val in zip(hmm_priors, cnn_priors):
        class_total = hmm_val/2 + cnn_val/2
        new_hmm = (hmm_val/2) / class_total
        new_cnn = (cnn_val/2) / class_total
        assert new_hmm + new_cnn == 1
        new_priors.append([new_hmm, new_cnn, 0, 0])
    return new_priors

def test(model):
    current_priors = numpy.array([[1/N_DISTRIBUTIONS], [1/N_DISTRIBUTIONS], [1/N_DISTRIBUTIONS], []])
    print(current_priors.shape)
    actual_priors = prior_distribution(fake_priors[0], fake_priors[1])
    print(numpy.array(actual_priors).shape)
    print("Probs:")
    probs = numpy.array(model.log_probability(current_priors, priors=actual_priors).tolist())
    print(probs)
    probs = F.softmax(probs)
    print("Softmax:")
    print(probs)
    return
    preds = [prob.argmax() for prob in probs]
    y_pred.extend(preds)

    print(classification_report(
       y_true=y_true,
       y_pred=y_pred,
       target_names=test_data.classes)
    )

if __name__ == '__main__':
    model = make_model()
    test(model)
