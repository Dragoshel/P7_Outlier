from cnn.cnn import CNN
from cnn.test import test_model
from cnn.train import train_model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# Define relevant variables for the ML task
learning_rate = 0.001

def create_cnn(num_epochs: int, train_loader: DataLoader, validate_loader: DataLoader, test_loader: DataLoader, no_norms: int) -> CNN:
    model = CNN(no_norms)
    # Whether or not to run on GPU (cuda) or CPU
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
    else:
        device= torch.device('cpu')

    # Set Loss function with criterion
    criterion = nn.CrossEntropyLoss()

    # Set optimizer with optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.05, momentum = 0.9)  

    print('Starting training of the model')
    train_model(num_epochs, train_loader, validate_loader, device, model, optimizer, criterion)

    print('Finished training, testing model')
    test_model(test_loader, device, model)
    
    return model