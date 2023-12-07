import argparse
import os
import random
import numpy
import pandas

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
    
    def __init__(self, classes, batch_size, no_epoch, model_folder, seed, accuracy="acc"):
        self.model_path = f'{model_folder}/cnn_model_{seed}_{len(classes)}_{batch_size}_{no_epoch}_{accuracy}.pth'
        print(f"[INFO] Loading CNN {self.model_path}")
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
            
        self.class_accuracies = {
            DataType.OUTLIER: {
                DataType.NORMAL: 0,
                DataType.NOVEL: 0,
                DataType.OUTLIER: 0,
                'total': 0
            },
            DataType.NOVEL: {
                DataType.NORMAL: 0,
                DataType.NOVEL: 0,
                DataType.OUTLIER: 0,
                'total': 0
            },
            DataType.NORMAL: {
                DataType.NORMAL: 0,
                DataType.NOVEL: 0,
                DataType.OUTLIER: 0,
                'total': 0
            }
        }
        self.accuracy_over_time = {
            DataType.NORMAL: [0]*20,
            DataType.NOVEL: [0]*20,
            DataType.OUTLIER: [0]*20,
            'all': [0]*20
        }
        self.total_accuracy = 0
    
    def _save_model(self):
        torch.save(self.model, self.model_path)
        
    def _load_model(self):
        return torch.load(self.model_path)
        
    def get_probabilities(self, images):
        images = images.to(torch.float32)
        probs = self.model(images)
        probs = torch.exp(probs.cpu())
        return probs
        
    def train(self):
        generator = torch.Generator()
        generator.manual_seed(10)
        #print('[INFO] Creating train and validation sets')
        full_set = Subset(self.train_data, [i for i, target in enumerate(self.train_data.targets) if target in self.normal_classes])
        
        train = int(len(full_set) * 0.8) # Always to a 4/5 to 1/5 split of set
        validate = int(len(full_set) - train)
        training_set, validation_set = random_split(full_set,
            [train, validate],
            generator=generator
        )
        
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True, generator=generator)
        validation_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False, generator=generator)

        #print('Training set has {} instances'.format(len(training_set)))
        #print('Validation set has {} instances'.format(len(validation_set)))
        
        # Set Loss function with criterion
        criterion = nn.CrossEntropyLoss()

        # Set optimizer with optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay = 0.05, momentum = 0.9)
        
        train_model(self.epochs, training_loader, validation_loader, self.model, optimizer, criterion)
        
    def test(self):
        generator = torch.Generator()
        generator.manual_seed(10)
        #print(f"[INFO] Initializing test run")
        test_data_subset = Subset(self.test_data, [i for i, target in enumerate(self.test_data.targets) if target in self.normal_classes])
        test_data_loader = DataLoader(test_data_subset,
            shuffle=True, batch_size=1000, generator=generator)
        
        #print(f"[INFO] Testing model with {len(self.test_data)} datapoints ...")
        correct = 0
        total = 0
        for i, (test_images, test_labels) in enumerate(test_data_loader):
            #print(f"[INFO] Running batch {i} of {len(test_data_loader)}")
            images = test_images.to(torch.float32)
            labels = torch.tensor(index_labels(test_labels))
            probs = self.model(images)
            probs = torch.exp(probs.cpu())
            preds = torch.tensor([prob.argmax() for prob in probs])
            correct += (preds == labels).sum().item()
            total += len(labels)
        
        #print(f"[INFO] Finished testing of the model")
        old_accuracy = self.accuracy
        self.accuracy = str(round(correct/total * 100, 2)).replace(".", "_")
        #print(f"[INFO] Accuracy score: {self.accuracy}%")
        if not os.path.exists(self.model_path.replace(old_accuracy, self.accuracy)):
            os.rename(self.model_path, self.model_path.replace(old_accuracy, self.accuracy))

    def classify(self, prob, batch, label):
        if prob >= 0.95:
            # Normal
            pred_label = DataType.NORMAL
        elif prob <= 0.6:
            # Outlier
            pred_label = DataType.OUTLIER
        else:
            # Novelty
            pred_label = DataType.NOVEL
        
        self.class_accuracies[label][pred_label] += 1
        self.class_accuracies[label]["total"] += 1
        self.total_accuracy += 1 if pred_label == label else 0
        
        self.accuracy_over_time[pred_label][batch] += 1 if pred_label == label else 0
        self.accuracy_over_time['all'][batch] += 1 if pred_label == label else 0

    def calculate_class_accuracies(self):
        for types in self.class_accuracies.values():
            if types["total"] != 0:
                types[DataType.NORMAL] = round(types[DataType.NORMAL]/types["total"], 4)
                types[DataType.NOVEL] = round(types[DataType.NOVEL]/types["total"], 4)
                types[DataType.OUTLIER] = round(types[DataType.OUTLIER]/types["total"], 4)

    def calculate_accuracy_for_batch(self, batch, no_normal, no_novel, no_outlier, total):
        self.accuracy_over_time[DataType.NORMAL][batch] = self._calculate_accuracy(batch, DataType.NORMAL, no_normal)
        self.accuracy_over_time[DataType.NOVEL][batch] = self._calculate_accuracy(batch, DataType.NOVEL, no_novel)
        self.accuracy_over_time[DataType.OUTLIER][batch] = self._calculate_accuracy(batch, DataType.OUTLIER, no_outlier)
        self.accuracy_over_time["all"][batch] = self._calculate_accuracy(batch, 'all', total)
        
    def _calculate_accuracy(self, batch, type, total):
        if total == 0 and batch != 0:
            return self.accuracy_over_time[type][batch-1]
        elif total == 0:
            return 0
        else:
            return round(self.accuracy_over_time[type][batch]/total, 2)
        
    def save_accuracy(self, extension, folder):
        if not os.path.exists(f"results_{folder}"):
            os.makedirs(f"results_{folder}")
            
        accuracy_data = pandas.DataFrame.from_dict(self.accuracy_over_time, orient="index")
        accuracy_data.to_csv(f"results_{folder}/cnn_accuracy_{extension}.csv")
        
        overall_accuracy = pandas.DataFrame.from_dict(self.class_accuracies)
        overall_accuracy.rename(columns={2: "Actual outlier", 1: "Actual Novel", 0: "Actual Normal"})
        overall_accuracy.to_csv(f"results_{folder}/cnn_class_accuracies_{extension}.csv")