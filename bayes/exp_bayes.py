from sklearn.metrics import classification_report
import torch

from dataloaders.test_loader import testing_data_loader

BATCH_SIZE = 1000

class HMM_buffer():
    buffer_img = []
    
    def __init__(self, models):
        self.hmms = models
        
    def __len__(self):
        return len(self.buffer_img)
    
    def add(self, image):
        self.buffer_img.append(image)
        
    def empty(self):
        for hmm in self.hmms:
            hmm.retrain(self.images)
        self.images = []
    
class Bayes():
    p_models = [0.7, 0.3]
    cnn_threshold = 0.95 * p_models[0]
    hmm_threshold = 0.90 * p_models[1]
    
    def __init__(self, normal, novel, pct_outliers, cnn, hmms):
        self.normal_classes = normal
        self.novel_classes = novel
        self.outliers = int(10000 / (1-pct_outliers) * pct_outliers)
        self.cnn = cnn
        self.hmms = hmms
        self.buffer = HMM_buffer(self.hmms)

    def run(self):
        test_loader, test_labels = testing_data_loader(BATCH_SIZE, self.outliers)

        y_true = []
        y_pred = []
        target_names = ['Normal', 'Novelty', 'Outlier']
        for images, labels in test_loader:
            cnn_probabilities = self.cnn.get_probabilities(images) * self.p_models[0]
            hmm_probabilities = self.hmms.get_all_probabilities(images) * self.p_models[1]

            for i in range(len(labels)):
                label = labels[i]

                cnn_probability = cnn_probabilities[i]
                hmm_probability = hmm_probabilities[i]

                cnn_max_probability = torch.max(cnn_probability)
                hmm_max_probability = torch.max(hmm_probability)

                original_label = test_labels[label] if label < len(test_labels) else label
                if original_label in self.normal_classes:
                    y_true.append(0)
                elif original_label in self.novelty_classes:
                    y_true.append(1)
                else:
                    y_true.append(2)

                if cnn_max_probability >= self.cnn_threshold:
                    # Normal
                    y_pred.append(0)
                elif hmm_max_probability >= self.hmm_threshold:
                    # Outlier
                    y_pred.append(2)
                else:
                    # Novelty
                    y_pred.append(1)
                    self.buffer.add(images[i])
                    
            if len(self.buffer) > 100:
                self.buffer.empty()

        print(classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names)
        )
