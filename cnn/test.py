
import time
from cnn.cnn import CNN
from torch import device
import torch

def test_model(test_loader: torch.utils.data.DataLoader, device: device, model: CNN, novel_labels: list) -> None:
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
        certainty_scores = {
            DataType.NORMAL: [],
            DataType.NOVEL: [],
            DataType.OUTLIER: []
        }
        correct = 0
        total = 0
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            probabilities = model(image)
            certainty, prediction = torch.max(output.data, 1)
            total += label.size(0)
            correct += (prediction == label).sum().item()
        
        totaltime = round((time.time() - starttime), 2)
        print('Accuracy of the network on the {} test images: {} %, Total time: {}'.format(, 100 * correct / total, totaltime))

def _update