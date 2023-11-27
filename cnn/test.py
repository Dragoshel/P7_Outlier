
import time
from cnn.cnn import CNN
from utils.data_types import DataType
import torch
from utils.classes import get_normal_classes, get_novel_classes, index_labels

def test_model(test_loader: torch.utils.data.DataLoader, model: CNN, labels: list) -> None:
    """Perform the testing of the models accuracy after finishing the training phase, during this no gradient descent
    is used, so no weights are adjusted.

    Args:
        batch_size (int): Amount of images to test with at a time
        model (CNN): Model to train
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
        normal_classes = get_normal_classes()
        novel_labels = get_novel_classes()
        for image, label in test_loader:  
            image = image.to(torch.float32)
            orig_label = [labels[label]] if label < len(labels) else label.tolist()
            label = torch.tensor(index_labels(orig_label))
            probabilities = model(image)
            readable_probs = torch.exp(probabilities.cpu())
            certainty, prediction = torch.max(readable_probs, 1)
            if orig_label[0] in novel_labels:
                certainty_scores[DataType.NOVEL].append(certainty)
            elif orig_label[0] in normal_classes:
                certainty_scores[DataType.NORMAL].append(certainty)
                # normal_total += label.size(0)
                normal_total += 1
                # normal_correct += (prediction == label).sum().item()
                if prediction == label:
                    normal_correct += 1
            else:
                certainty_scores[DataType.OUTLIER].append(certainty)
                
        totaltime = round((time.time() - starttime), 2)
        print(f'Total time for testing: {totaltime}')
        print(f'Accuracy on normal data: {normal_correct/normal_total*100}%')
        print(f'Average accuracy on normal data: {sum(certainty_scores[DataType.NORMAL])/len(certainty_scores[DataType.NORMAL])*100}%')
        print(f'Average accuracy on novel data: {sum(certainty_scores[DataType.NOVEL])/len(certainty_scores[DataType.NOVEL])*100}%')
        print(f'Average accuracy on outlier data: {sum(certainty_scores[DataType.OUTLIER])/len(certainty_scores[DataType.OUTLIER])*100}%')
        print(f'Amount of normal: {len(certainty_scores[DataType.NORMAL])}')
        print(f'Amount of novel: {len(certainty_scores[DataType.NOVEL])}')
        print(f'Amount of outlier: {len(certainty_scores[DataType.OUTLIER])}')
