import random
import torch
import os

from cnn.create_cnn import create_cnn
from cnn.test import test_model
from dataloaders.test_loader import testing_data_loader
from dataloaders.train_loader import training_data_loaders
from utils.classes import pick_classes, get_normal_classes

# Counteract non-determinism
random.seed(10)
torch.manual_seed(10)

# Pick normal, abnormal and novelty class labels
no_norms = 8
pct_outliers = 0.2
no_outliers = int(50000 / (1-pct_outliers) * pct_outliers)

# Set train and testdata size and run throughs
batch_size = 64
num_epochs = 10
data_path = './data'

model_path = 'data/model.pth'

if __name__ == '__main__':
    print('Initialising...')
    pick_classes(no_norms)
    if torch.cuda.is_available():
        print('Using GPU')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device= torch.device('cpu')

    if not os.path.exists(model_path):
        print('Creating train and validation sets')
        training_loader, validation_loader = training_data_loaders(batch_size, data_path)
        
        cnn_model, _ = create_cnn(num_epochs, training_loader, validation_loader, device, no_norms)
        print('Finished training, saving model...')
        torch.save(cnn_model, model_path)

    cnn_model = torch.load(model_path).to(device)
    test_loader = testing_data_loader(1, data_path, no_outliers)
    test_model(test_loader, device, cnn_model)
