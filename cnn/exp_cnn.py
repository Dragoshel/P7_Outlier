import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import PILToTensor

from cnn.train import train_model
from utils.classes import index_labels
from torch.utils.data import Subset

import torch.nn as nn

from utils.data_types import DataType

# Define relevant variables for the ML task
learning_rate = 0.001

class _CNN(nn.Module):
    def __init__(self, classes):
        super(_CNN, self).__init__()
        k = 3
                
        self.first_feat_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=k),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(0.25)
        )
        
        self.second_feat_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=k),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout2d(0.25)
        )
        
        self.fully_conn = nn.Sequential(
            nn.Flatten(),
            # 64*2*2*2*2 (A 2 per convolutional layer, 64 is the channels from the last convolutional layer)
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(classes)),
            # Function for use with multi class clasification
            nn.LogSoftmax(dim=1)
        )
    
    # Progresses data across layers
    def forward(self, x):
        out = self.first_feat_layer(x)
        out = self.second_feat_layer(out)    
        
        out = self.fully_conn(out)
        return out

class CNN_model():
    train_data = MNIST(root='data/cnn/training', train=True, download=True, transform=PILToTensor())
    test_data = MNIST(root='data/cnn/testing', train=False, download=True, transform=PILToTensor())
    outlier_data = FashionMNIST(root='data/cnn/outlier', train=True, download=True, transform=PILToTensor())
    
    generator = torch.Generator()
    generator.manual_seed(10)
    
    def __init__(self, classes, batch_size, no_epoch, model_folder, accuracy="0"):
        self.model_path = f'{model_folder}/cnn_model_{len(classes)}_{batch_size}_{no_epoch}_{accuracy}.pth'
        self.batch_size = batch_size
        self.epochs = no_epoch
        self.normal_classes = classes
        self.accuracy = accuracy
        
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        
        if not os.path.exists(self.model_path):
            self.model = _CNN(classes)
            self.train()
            self._save_model()
            self.test()
        else:
            self.model = self._load_model()
            self.model.eval()
    
    def _save_model(self):
        torch.save(self.model, self.model_path)
        
    def _load_model(self):
        self.model = torch.load(self.model_path)
        
    def train(self):
        print('[INFO] Creating train and validation sets')
        full_set = Subset(self.train_data, [i for i, target in enumerate(self.train_data.targets) if target in self.normal_classes])
        
        train = int(len(full_set) * 0.8) # Always to a 4/5 to 1/5 split of set
        validate = int(len(full_set) - train)
        training_set, validation_set = random_split(full_set,
            [train, validate],
            generator=self.generator
        )
        
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True, generator=self.generator)
        validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False, generator=self.generator)

        print('Training set has {} instances'.format(len(training_set)))
        print('Validation set has {} instances'.format(len(validation_set)))
        
        # Set Loss function with criterion
        criterion = nn.CrossEntropyLoss()

        # Set optimizer with optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay = 0.05, momentum = 0.9)
        
        train_model(self.epochs, training_loader, validation_loader, self.model, optimizer, criterion)
        
    def test(self):
        print(f"[INFO] Initializing test run")
        test_data_loader = DataLoader(self.test_data,
            shuffle=True, batch_size=1000, generator=self.generator)
        
        print(f"[INFO] Testing model with {len(self.test_data)} datapoints ...")
        correct = 0
        total = 0
        for i, (test_images, test_labels) in enumerate(test_data_loader):
            print(f"[INFO] Running batch {i} of {len(test_data_loader)}")
            images = test_images.to(torch.float32)
            labels = torch.tensor(index_labels(test_labels))
            probs = self.model(images)
            probs = torch.exp(probs.cpu())
            preds = torch.tensor([prob.argmax() for prob in probs])
            correct += (preds == labels).sum().item()
            total += len(labels)
        
        print(f"[INFO] Finished testing of the model")
        old_accuracy = self.accuracy
        self.accuracy = str(round(correct/total * 100))
        print(f"[INFO] Accuracy score: {self.accuracy}%")
        if not os.path.exists(self.model_folder.replace(old_accuracy, self.accuracy)):
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

        for images, _ in test_loader:
            pred = self.model(images)
            probs = F.softmax(pred, dim=1)

            for prob in probs:
                max_prob = torch.max(prob)
                max_prob = int(max_prob * no_thresholds)
                thresholds[max_prob] += 1
                
        print('CNN Thresholds:')
        for threshold in thresholds:
            print('{:.2f}'.format(threshold / test_data_size), end=', ')