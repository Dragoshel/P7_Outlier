
import time
from cnn.cnn import CNN
from loaders.data_loader import testing_data_loader
from torch import device
import torch

def test_model(batch_size: int, device: device, model: CNN, data_path: str) -> None:
    """Perform the testing of the models accuracy after finishing the training phase, during this no gradient descent
    is used, so no weights are adjusted.

    Args:
        batch_size (int): Amount of images to test with at a time
        model (CNN): Model to train
        device (device): Configured device for running the training, either GPU or CPU
        data_path (str): Path for saving the data for training and validation
    """    
    test_loader = testing_data_loader(batch_size, data_path)
    
    starttime = time.time()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        totaltime = round((time.time() - starttime), 2)
        print('Accuracy of the network on the {} test images: {} %, Total time: {}'.format(50000, 100 * correct / total, totaltime))
