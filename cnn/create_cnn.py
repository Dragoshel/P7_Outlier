from cnn.cnn import CNN
from cnn.test import test_model
from cnn.train import train_model
import torch
import torch.nn as nn

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10
data_path = './cnn_data'

def create_cnn() -> CNN:
    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNN(num_classes)
    if 'cuda':
        model.cuda()

    # Set Loss function with criterion
    criterion = nn.CrossEntropyLoss()

    # Set optimizer with optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.05, momentum = 0.9)  

    print('Starting training of the model')
    train_model(batch_size, num_epochs, device, model, data_path, optimizer, criterion)

    print('Finished training, testing model')
    test_model(batch_size, device, model, data_path)
    
    return model