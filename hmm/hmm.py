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


# Model distribution initialisation parameters
N_OBSERVATIONS = 256 # fixed
PREFERRED_SUM = 0.8
N_DISTRIBUTIONS = 30

N_DIMENSIONS = 28 * 28
# num of training iternations hmm will do for every batch
N_FIT_ITER = 100
# train data % subset
N_TRAIN_DATA = 0.50

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"[INFO] Running on {device.type}...")

print("[INFO] Loading the MNIST Dataset ...")
train_data = MNIST(root='training', train=True,
                   download=True, transform=_transform)
test_data = MNIST(root='testing', train=False,
                  download=True, transform=_transform)


def create_distributions(num_dist, pref_sum, count):
    global device
    uniform_dist = Categorical([(numpy.ones(count) / count)]).to(device)
    dists = [uniform_dist]
    for i in range(num_dist):
        pref_size = int(count / 2 ** i)
        unpref_size = count - pref_size
        pref_part = numpy.array([pref_sum] * pref_size) / pref_size
        unpref_part = numpy.array([1 - pref_sum] * unpref_size) / unpref_size
        dist = Categorical([numpy.concatenate([unpref_part, pref_part])]).to(device)
        dists.append(dist)
    return dists


def train_model(digit):
    global device
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
            train_data_subset, shuffle=True, batch_size=1000, generator=generator)

        # Setting up the base model
        distributions = create_distributions(
            N_DISTRIBUTIONS, PREFERRED_SUM, N_OBSERVATIONS)
        model = DenseHMM(distributions, max_iter=N_FIT_ITER, verbose=True)
        model = model.to(device)

        # Train model to fit sequences observed in a single number
        print(f"[INFO] Fitting model for digit {digit} ...")
        for train_images in train_data_loader:
            train_images = train_images.reshape(-1, N_DIMENSIONS, 1)
            train_images = train_images.to(torch.int64).to(device)
            
            
            model.fit(train_images)
            # model.summarize(train_images)
            print(f"[INFO] Finished batch ...")

        print(f"[INFO] From summaries on digit {digit} ...")
        # model.from_summaries()

        torch.save(model, model_path)


def test(models):
    global device
    test_data_loader = DataLoader(test_data.data, shuffle=True, batch_size=100)
    
    print(f"[INFO] Testing model with {len(test_data)} datapoints ...")
    y_true = test_data.targets
    y_pred = []
    for i, test_images in enumerate(test_data_loader):
        print(f"Batch {i}")
        test_images = test_images.reshape(-1, N_DIMENSIONS, 1)
        test_images = test_images.to(torch.int64).to(device)

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

def parse():
    global device
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
    global device
    args = parse()

    if args.test:
        models = []
        for digit in range(args.num_classes):
            # model = torch.load(f'output/model1.pth')
            model = torch.load(f'output/model{digit}.pth').to(device)
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