from pomegranate.distributions import Normal, Categorical
from pomegranate.hmm import DenseHMM, SparseHMM

from torchvision.transforms import transforms, ToTensor
from torchvision.datasets import KMNIST, MNIST
from torch.utils.data import random_split, DataLoader, Subset

import sys
print(sys.path)
import torch.nn.functional as F

from sklearn.metrics import classification_report

import argparse
import torch
import numpy
import random
import math
import os

from cnn.priors_cnn import load_cnn, priors_cnn
from hmm.priors_hmm import load_hmms, priors_hmms


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
# Setup values

NO_CLASSES = 10
P_CLASS = 1 / NO_CLASSES


test_data = MNIST(root='data/hmm/testing', train=False,
                  download=True, transform=_transform)

fake_priors = [[0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8],
               [0.2, 0.8]]
 
def calculate_bayes(hmm_priors: list, cnn_priors):
    # Calculate probability of each class given the models values
    p_models = []
    for hmm_val, cnn_val in zip(hmm_priors, cnn_priors):
        class_prob = hmm_val/P_CLASS + cnn_val/P_CLASS
        p_models.append(class_prob)
    return p_models

def test(cnn, hmm):
    test_data_loader = DataLoader(test_data.data, shuffle=False, batch_size=1000, generator=generator)
    
    print(f"[INFO] Testing model with {len(test_data)} datapoints ...")
    y_true = test_data.targets
    y_pred = []
    for i, test_images in enumerate(test_data_loader):
        print(f"Batch {i}")
        for image in test_images:
            image = image.to(torch.float32)
            image = image.reshape(1, 1, 28, 28)
            cnn_probs = priors_cnn(cnn, image)
            hmm_probs = priors_hmms(hmm, image)
            for cnn_prob, hmm_prob in zip(cnn_probs, hmm_probs):
                bayes_prob = calculate_bayes(hmm_prob, cnn_prob)
                preds = [prob.argmax() for prob in bayes_prob]
                y_pred.extend(preds)
    print(classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=test_data.classes)
    )

if __name__ == "__main__":
    cnn = load_cnn("./models")
    hmm = load_hmms("./models", NO_CLASSES)
    test(cnn, hmm)

