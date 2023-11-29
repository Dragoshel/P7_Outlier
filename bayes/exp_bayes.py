import os
from sklearn.metrics import classification_report
import torch

from dataloaders.test_loader import testing_data_loader
from utils.data_types import DataType
import pandas

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
    
    class_accuracies = {
        DataType.OUTLIER: {
            DataType.NORMAL: 0,
            DataType.NOVEL: 0,
            DataType.OUTLIER: 0,
            'total': 0
        },
        DataType.NOVEL: {
            DataType.NORMAL: 0,
            DataType.NOVEL: 0,
            DataType.OUTLIER: 0,
            'total': 0
        },
        DataType.NORMAL: {
            DataType.NORMAL: 0,
            DataType.NOVEL: 0,
            DataType.OUTLIER: 0,
            'total': 0
        }
    }
    accuracy_over_time = {
        DataType.NORMAL: [],
        DataType.NOVEL: [],
        DataType.OUTLIER: [],
        'All': []
    }
    
    def __init__(self, normal, novel, pct_outliers, cnn, hmms):
        self.normal_classes = normal
        self.novel_classes = novel
        self.outliers = int(10000 / (1-pct_outliers) * pct_outliers)
        self.cnn = cnn
        self.hmms = hmms
        self.buffer = HMM_buffer(self.hmms)
        
    def _calculate_class_accuracies(self):
        for types in self.class_accuracies.values():
            types[DataType.NORMAL] = round(types[DataType.NORMAL]/types["total"], 2)
            types[DataType.NOVEL] = round(types[DataType.NOVEL]/types["total"], 2)
            types[DataType.OUTLIER] = round(types[DataType.OUTLIER]/types["total"], 2)

    def run(self):
        test_loader, test_labels = testing_data_loader(self.outliers)
        for batch, (images, labels) in enumerate(test_loader):
            print(f"[INFO] Running batch {i} of {len(test_loader)}")
            no_outlier = 0
            no_novel = 0
            no_normal = 0
            
            cnn_probabilities = self.cnn.get_probabilities(images) * self.p_models[0]
            hmm_probabilities = self.hmms.get_all_probabilities(images) * self.p_models[1]
            pred_labels = []
            for i in range(len(labels)):
                label = labels[i]

                cnn_probability = cnn_probabilities[i]
                hmm_probability = hmm_probabilities[i]

                cnn_max_probability = torch.max(cnn_probability)
                hmm_max_probability = torch.max(hmm_probability)

                original_label = test_labels[label] if label < len(test_labels) else label
                if original_label in self.normal_classes:
                    # Normal
                    actual_label = DataType.NORMAL
                    no_normal += 1
                elif original_label in self.novelty_classes:
                    # Novel
                    actual_label = DataType.NOVEL
                    no_novel += 1
                else:
                    # Outlier
                    actual_label = DataType.OUTLIER
                    no_outlier += 1
                
                labels[i] = actual_label
                
                if cnn_max_probability >= self.cnn_threshold:
                    # Normal
                    pred_label = DataType.NORMAL
                elif hmm_max_probability >= self.hmm_threshold:
                    # Outlier
                    pred_label = DataType.OUTLIER
                else:
                    # Novelty
                    pred_label = DataType.NOVEL
                    self.buffer.add(images[i])
                pred_labels.append(pred_label)
                
                self.class_accuracies[actual_label][pred_label] += 1
                self.class_accuracies[actual_label]["total"] += 1
                
                self.accuracy_over_time[pred_label][batch] += 1 if pred_label == actual_label else 0
                
            self.accuracy_over_time[DataType.NORMAL][batch] = round(self.accuracy_over_time[DataType.NORMAL][batch]/no_normal, 2)
            self.accuracy_over_time[DataType.NOVEL][batch] = round(self.accuracy_over_time[DataType.NOVEL][batch]/no_novel, 2)
            self.accuracy_over_time[DataType.OUTLIER][batch] = round(self.accuracy_over_time[DataType.OUTLIER][batch]/no_outlier, 2)
            total_correct = (torch.tensor(pred_labels) == torch.tensor(labels)).sum().item()
            self.accuracy_over_time["all"][batch] = round(total_correct/len(labels), 2)
            
            if len(self.buffer) > 50:
                self.buffer.empty()
        
        self._calculate_class_accuracies()
    
    def save_accuracy(self, extension):
        if not os.path.exists("results"):
            os.makedirs("results")
            
        accuracy_data = pandas.DataFrame.from_dict(self.accuracy_over_time, orient="index")
        accuracy_data.to_csv(f"results/accuracy_{extension}.csv")
        
        overall_accuracy = pandas.DataFrame.from_dict(self.class_accuracies)
        overall_accuracy.to_csv(f"results/class_accuracies_{extension}.csv")
