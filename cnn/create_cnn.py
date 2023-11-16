import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn.cnn import CNN
from cnn.train import train_model

# Define relevant variables for the ML task
learning_rate = 0.001

def create_cnn(num_epochs: int, train_loader: DataLoader, validate_loader: DataLoader, no_normal) -> tuple:
    model = CNN(no_normal)

    # Set Loss function with criterion
    criterion = nn.CrossEntropyLoss()

    # Set optimizer with optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.05, momentum = 0.9)  

    print('Starting training of the model')
    train_model(num_epochs, train_loader, validate_loader, model, optimizer, criterion)
    
    return model, device