
import time
from cnn.cnn import CNN
from utils.data_types import DataType
from torch import device
import torch

def test_model(test_loader: torch.utils.data.DataLoader, device: device, model: CNN, novel_labels: list, normal_classes: list) -> None:
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
        normal_correct = 0
        normal_total = 0
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            probabilities = model(image)
            certainty, prediction = torch.max(probabilities.data, 1)
            if prediction in novel_labels:
                certainty_scores[DataType.NOVEL].append(certainty)
            elif prediction in normal_classes:
                certainty_scores[DataType.NORMAL].append(certainty)
                normal_total += label.size(0)
                normal_correct += (prediction == label).sum().item()
            else:
                certainty_scores[DataType.OUTLIER].append(certainty)
            
        totaltime = round((time.time() - starttime), 2)
        print(f'Total time for testing: {totaltime}')
        print(f'Accuracy on normal data: {normal_correct/normal_total*100}%')
        print(f'Average accuracy on normal data: {sum(certainty_scores[DataType.NORMAL])/len(certainty_scores[DataType.NORMAL])*100}%')
        print(f'Average accuracy on novel data: {sum(certainty_scores[DataType.NOVEL])/len(certainty_scores[DataType.NOVEL])*100}%')
        print(f'Average accuracy on outlier data: {sum(certainty_scores[DataType.OUTLIER])/len(certainty_scores[DataType.OUTLIER])*100}%')
