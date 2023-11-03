import random
import torch
import os

from cnn.create_cnn import create_cnn
from cnn.test import test_model
from dataloaders.test_loader import testing_data_loader
from dataloaders.train_loader import training_data_loaders

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


def pick_normal_classes(no_normal: int) -> tuple:
    mnist_classes = range(0, 10)
    normal_classes = random.sample(mnist_classes, no_normal)
    novel_classes = [no for no in mnist_classes if no not in normal_classes]
    print('Normal classes: {}, Novelty classes: {}'.format(normal_classes, novel_classes))
    return normal_classes, novel_classes

if __name__ == '__main__':
    print('Initialising...')
    normal_classes, novel_classes = pick_normal_classes(no_norms)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device= torch.device('cpu')

    if not os.path.exists(model_path):
        print('Creating data sets')
        training_loader, validation_loader = training_data_loaders(batch_size, data_path, normal_classes)
        
        print('Creating CNN')
        cnn_model, _ = create_cnn(num_epochs, training_loader, validation_loader, device)

        torch.save(cnn_model, model_path)

    cnn_model = torch.load(model_path).to(device)
    test_loader = testing_data_loader(1, data_path, no_outliers)
    test_model(test_loader, device, cnn_model, novel_classes, normal_classes)