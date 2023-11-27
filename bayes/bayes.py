import random

import numpy
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataloaders.test_loader import testing_data_loader
from hmm.grid_hmm import reform_images

BATCH_SIZE = 10
PCT_OUTLIERS = 0.1
NO_OUTLIERS = int(50000 / (1-PCT_OUTLIERS) * PCT_OUTLIERS)

random.seed(10)
torch.manual_seed(10)

MODEL_PATH = 'models'
DATA_PATH = 'data'

def cnn_prob(model, images):
    probs = model(images)
    probs = torch.exp(probs)
    return probs

def hmm_prob(models, images):
    images = images.reshape(-1, 28 * 28, 1)
    images = images.to(torch.int64)

    images = reform_images(images)

    probs = [model.log_probability(images).tolist() for model in models]
    probs = numpy.array(probs).transpose()
    probs = torch.tensor(probs)
    probs = F.softmax(probs, dim=1)

    return probs
# CNN 96%

# 0.53 0.81
# 0.61 0.76
# 0.

def bayes(normal_classes, novelty_classes, cnn_threshold=0.672 , hmm_threshold=0.285):
    test_loader, test_labels = testing_data_loader(BATCH_SIZE, DATA_PATH, NO_OUTLIERS)

    cnn_model = torch.load(f'{MODEL_PATH}/cnn_model.pth')
    cnn_model.eval()

    hmm_models = []
    for digit in normal_classes:
        model = torch.load(f'{MODEL_PATH}/model{digit}.pth')
        hmm_models.append(model)

    y_true = []
    y_pred = []
    target_names = ['Normal', 'Novelty', 'Outlier']
    loader = tqdm(test_loader, 'Processing Data')
    for images, labels in loader:
        cnn_probabilities = cnn_prob(cnn_model, images)
        hmm_probabilities = hmm_prob(hmm_models, images)


        for i in range(len(labels)):
            label = labels[i]

            cnn_probability = cnn_probabilities[i]
            hmm_probability = hmm_probabilities[i]


            cnn_max_probability = torch.max(cnn_probability)
            hmm_max_probability = torch.max(hmm_probability)

            original_label = test_labels[label] if label < len(test_labels) else label
            if original_label in normal_classes:
                y_true.append(0)
            elif original_label in novelty_classes:
                y_true.append(1)
                print(f'CNN: {cnn_probability}')
                print(f'HMM: {hmm_probability}')
            else:
                y_true.append(2)

            if cnn_max_probability >= cnn_threshold:
                y_pred.append(0)
            elif hmm_max_probability >= hmm_threshold:
                y_pred.append(2)
            else:
                y_pred.append(1)

        # return
    print(classification_report(
       y_true=y_true,
       y_pred=y_pred,
       target_names=target_names)
    )
