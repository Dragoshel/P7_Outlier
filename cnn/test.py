
import time
from cnn.cnn import CNN
from torch import device
import torch

from utils.data_types import DataType

def test_model(test_loader: torch.utils.data.DataLoader, device: device, model: CNN) -> None:
    """Perform the testing of the models accuracy after finishing the training phase, during this no gradient descent
    is used, so no weights are adjusted.

    Args:
        batch_size (int): Amount of images to test with at a time
        model (CNN): Model to train
        device (device): Configured device for running the training, either GPU or CPU
        data_path (str): Path for saving the data for training and validation
    """
    starttime = time.time()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:            
            labels = torch.tensor([DataType.NORMAL] * len(labels))
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        totaltime = round((time.time() - starttime), 2)
        print('Accuracy of the network on the {} test images: {} %, Total time: {}'.format(50000, 100 * correct / total, totaltime))
