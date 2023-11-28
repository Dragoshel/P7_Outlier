from pomegranate.distributions import Categorical
from pomegranate.hmm import DenseHMM

from torchvision.transforms import PILToTensor
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import random_split, DataLoader, Subset

import torch.nn.functional as F

import torch
import numpy
import os

from utils.data_types import DataType

class HMM:
    img_size = 28
    n_dimensions = img_size **2
    preferred_sum = 0.8
    
    generator = torch.Generator()
    generator.manual_seed(10)
    
    def __init__(self, model_folder, pixel_per_square, observations, distributions, digit, load_model=False):
        self.pixel_per_square = pixel_per_square
        self.observations = observations
        self.distributions = distributions
        self.digit = digit
        self.model_path = f"{model_folder}/model{digit}.pth"
        
        if load_model:
            self._load_model()
        else:
            distributions = self._create_distributions()
            self.model = DenseHMM(distributions, verbose=True)
        
    def _make_grid(self, image):
        image = torch.flatten(image)
        image = torch.reshape(image, [self.img_size,self.img_size])
        h,w = image.shape
        h =(h // self.pixel_per_square) * self.pixel_per_square
        w = (w // self.pixel_per_square) * self.pixel_per_square
        image = image[:h, :w]
        image = image.reshape(h // self.pixel_per_square, 
                              self.pixel_per_square, 
                              -1, 
                              self.pixel_per_square).swapaxes(1, 2).reshape(
                                  h // self.pixel_per_square, w // self.pixel_per_square, -1).sum(
                                      axis=-1) % self.observations
        return image

    def _reform_images(self, images):
        new_images = []
        grid_sq = 28 // self.pixel_per_square
        for image in images:
            new_image = self._make_grid(image)
            new_image = new_image.reshape(grid_sq**2, 1)
            new_images.append(new_image.tolist())
        return torch.tensor(new_images)

    def _create_distributions(self):
        uniform_dist = Categorical([(numpy.ones(self.observations) / self.observations)])
        dists = [uniform_dist]
        for i in range(self.distributions):
            pref_size = int(self.observations / 2 ** i)
            unpref_size = self.observations - pref_size
            pref_part = numpy.array([self.preferred_sum] * pref_size) / pref_size
            unpref_part = numpy.array([1 - self.preferred_sum] * unpref_size) / unpref_size
            dist = Categorical([numpy.concatenate([unpref_part, pref_part])])
            dists.append(dist)
        return dists

    def train_model(self, train_data, fit_size):
        train_data_subset = [img for img, label in zip(
            train_data.data, train_data.targets) if label == self.digit]
        
        discard = int(len(train_data_subset) - fit_size)
        train_data_subset, _ = random_split(train_data_subset,
            [fit_size, discard],
            generator=self.generator
        )

        train_data_loader = DataLoader(
            train_data_subset, shuffle=True, batch_size=len(train_data_subset), generator=self.generator)
        print('Training set has {} instances'.format(len(train_data_subset)))
        
        # Train model to fit sequences observed in a single number
        print(f"[INFO] Fitting model for digit {self.digit} ...")
        for train_images in train_data_loader:
            # Reshape images to match (batch_size, 784, 1) with int values
            train_images = train_images.reshape(-1, self.n_dimensions, 1)
            train_images = train_images.to(torch.int64)
            # Format images to grids
            train_images = self._reform_images(train_images)

            self.model.fit(train_images)
    
    def save_model(self):
        torch.save(self.model, self.model_path)
        
    def _load_model(self):
        self.model = torch.load(self.model_path)
        
    def get_probabilities(self, images):
        # Reshape images to match (batch_size, 784, 1) with int values
        images = images.reshape(-1, 28**2, 1)
        images = images.to(torch.int64)
        # Format images to grids
        images = self._reform_images(images)

        return self.model.log_probability(images).tolist()
        

class HMM_Models:
    train_data = MNIST(root='data/hmm/training', train=True, download=True, transform=PILToTensor())
    test_data = MNIST(root='data/hmm/testing', train=False, download=True, transform=PILToTensor())
    outlier_data = FashionMNIST(root='data/hmm/outlier', train=True, download=True, transform=PILToTensor())
    models = []
    accuracy = 0
    
    generator = torch.Generator()
    generator.manual_seed(10)

    def __init__(self, fit_set_size, distributions, observations, pixels_per_square, accuracy=""):
        self.model_folder = f"models_{distributions}_{observations}_{fit_set_size}_{pixels_per_square}_{accuracy}"
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            
            for digit in range(10):
                print(f"[INFO] Initializing model for digit {digit} ...")
                hmm_for_digit = HMM(self.model_folder, pixels_per_square, observations, distributions, digit)
                hmm_for_digit.train_model(self.train_data, fit_set_size)
                hmm_for_digit.save_model()
                self.models.append(hmm_for_digit)
                
            print(f"[INFO] Finished creating models")
            self.all_class_test()
        else:
            print(f"[INFO] Loading model from folder ...")
            for digit in range(10):
                hmm_for_digit = HMM(self.model_folder, pixels_per_square, observations, distributions, digit, True)
                self.models.append(hmm_for_digit)
        
    def all_class_test(self):
        print(f"[INFO] Initializing test run")
        test_data_loader = DataLoader(self.test_data,
            shuffle=True, batch_size=1000, generator=self.generator)
        
        print(f"[INFO] Testing model with {len(self.test_data)} datapoints ...")
        correct = 0
        total = 0
        for i, (test_images, test_labels) in enumerate(test_data_loader):
            print(f"[INFO] Running batch {i} of {len(test_data_loader)}")
            probs = numpy.array([model.get_probabilities(test_images) for model in self.models])
            probs = probs.transpose()
            preds = torch.tensor([prob.argmax() for prob in probs])
            correct += (preds == test_labels).sum().item()
            total += len(test_labels)
            print(correct)
        
        print(f"[INFO] Finished testing of the models")
        old_accuracy = self.accuracy
        self.accuracy = round(correct/total * 100)
        print(f"[INFO] Accuracy score: {self.accuracy}%")
        if not os.path.exists(self.model_folder):
            os.rename(self.model_folder, self.model_folder.replace(old_accuracy, self.accuracy))

    def threshold(self, type=DataType.NORMAL, classes=[0,1,2,3,4,5,6,7,8,9], test_data_size=5000, no_thresholds=20):
        if type == DataType.NORMAL or type == DataType.NOVEL:
            test_data = self.test_data
        else:
            test_data = self.outlier_data
        test_data = Subset(test_data, [i for i, target in enumerate(test_data.targets) if target in classes])

        discard = len(test_data) - test_data_size
        test_data_subset, _ = random_split(test_data,
            [test_data_size, discard],
            generator=self.generator
        )
        
        test_loader = DataLoader(test_data_subset, batch_size=1000, shuffle=True, generator=self.generator)
        print('Threshold set has {} instances'.format(len(test_data_subset)))
        thresholds = [0] * (no_thresholds + 1)

        for test_images, _ in test_loader:
            probs = numpy.array([model.get_probabilities(test_images).tolist() for model in self.models])
            probs = torch.tensor(probs.transpose())
            probs = F.softmax(probs, dim=1)

            for prob in probs:
                max_prob = torch.max(prob)
                max_prob = int(max_prob * no_thresholds)
                thresholds[max_prob] += 1
        print('HMM Thresholds:')
        for threshold in thresholds:
            print('{:.2f}'.format(threshold / test_data_size), end=', ')