import math
import os
import random
import torch
import pandas

from dataloaders.test_loader import testing_data_loader
from utils.data_types import DataType
from torch.utils.data import random_split, DataLoader, Subset
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
from torchvision.datasets import MNIST, FashionMNIST


class HMM_buffer():
    
    def __init__(self, models):
        self.hmms = models
        self.buffer_img = []
        
    def __len__(self):
        return len(self.buffer_img)
    
    def add(self, image):
        self.buffer_img.append(image)
        
    def empty(self):
        all_images = torch.cat(self.buffer_img, dim=0)
        for hmm in self.hmms.models:
            hmm.retrain(all_images)
        self.buffer_img = []
    
class Bayes():
    train_data = MNIST(root='data/bayes/training', train=True, download=True, transform=PILToTensor())
    outlier_data = FashionMNIST(root='data/bayes/outlier', train=True, download=True, transform=PILToTensor())
    
    p_models = [0.7, 0.3]
    
    def __init__(self, normal, novel, pct_outliers, cnn, hmms):
        self.normal_classes = normal
        self.novel_classes = novel
        self.outliers = int(10000 * pct_outliers)
        self.cnn = cnn
        self.hmms = hmms
        self.buffer = HMM_buffer(self.hmms)
        
        self.class_accuracies = {
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
        self.accuracy_over_time = {
            DataType.NORMAL: [0]*20,
            DataType.NOVEL: [0]*20,
            DataType.OUTLIER: [0]*20,
            'all': [0]*20
        }
        
        self.buffer_batches = []
        
    def _calculate_class_accuracies(self):
        for types in self.class_accuracies.values():
            if types["total"] != 0:
                types[DataType.NORMAL] = round(types[DataType.NORMAL]/types["total"], 4)
                types[DataType.NOVEL] = round(types[DataType.NOVEL]/types["total"], 4)
                types[DataType.OUTLIER] = round(types[DataType.OUTLIER]/types["total"], 4)
            
    def _calculate_accuracy(self, batch, type, total):
        if total == 0 and batch != 0:
            return self.accuracy_over_time[type][batch-1]
        elif total == 0:
            return 0
        else:
            return round(self.accuracy_over_time[type][batch]/total, 2)

    def run(self, has_buffer):
        test_loader, test_labels = testing_data_loader(self.outliers)
        total_accuracy = 0
        total_data = 0
        for batch, (images, labels) in enumerate(test_loader):
            no_outlier = 0
            no_novel = 0
            no_normal = 0
            total = 0
            
            cnn_probabilities = self.cnn.get_probabilities(images)
            hmm_probabilities = self.hmms.get_all_probabilities(images)
            for i in range(len(labels)):
                label = labels[i]

                cnn_probability = cnn_probabilities[i]
                hmm_probability = hmm_probabilities[i]

                cnn_max_probability = torch.max(cnn_probability)
                hmm_probability = torch.nan_to_num(hmm_probability, 0.0)
                hmm_max_probability = torch.max(hmm_probability)
                diff_max_probability = torch.max(abs(cnn_probability-hmm_probability))

                original_label = test_labels[label] if label < len(test_labels) else label
                if original_label in self.normal_classes:
                    # Normal
                    actual_label = DataType.NORMAL
                    no_normal += 1
                elif original_label in self.novel_classes:
                    # Novel
                    actual_label = DataType.NOVEL
                    no_novel += 1
                else:
                    # Outlier
                    actual_label = DataType.OUTLIER
                    no_outlier += 1
                
                labels[i] = actual_label
                total += 1
                
                if cnn_max_probability >= 0.95 or diff_max_probability <= 0.05:
                    # Normal
                    pred_label = DataType.NORMAL
                elif hmm_max_probability <= 0.3 or (hmm_max_probability > 0.7 and diff_max_probability >= 0.2) or cnn_max_probability < 0.6:
                    # Outlier
                    pred_label = DataType.OUTLIER
                else:
                    # Novelty
                    pred_label = DataType.NOVEL
                    self.buffer.add(images[i])
                    
                self.cnn.classify(cnn_max_probability, batch, actual_label)
                self.hmms.classify(hmm_max_probability, batch, actual_label)
                
                self.class_accuracies[actual_label][pred_label] += 1
                self.class_accuracies[actual_label]["total"] += 1
                
                self.accuracy_over_time[pred_label][batch] += 1 if pred_label == actual_label else 0
                self.accuracy_over_time['all'][batch] += 1 if pred_label == actual_label else 0
                total_accuracy += 1 if pred_label == actual_label else 0
            
            self.accuracy_over_time[DataType.NORMAL][batch] = self._calculate_accuracy(batch, DataType.NORMAL, no_normal)
            self.accuracy_over_time[DataType.NOVEL][batch] = self._calculate_accuracy(batch, DataType.NOVEL, no_novel)
            self.accuracy_over_time[DataType.OUTLIER][batch] = self._calculate_accuracy(batch, DataType.OUTLIER, no_outlier)
            self.accuracy_over_time["all"][batch] = self._calculate_accuracy(batch, 'all', total)
            self.hmms.calculate_accuracy_for_batch(batch, no_normal, no_novel, no_outlier, total)
            self.cnn.calculate_accuracy_for_batch(batch, no_normal, no_novel, no_outlier, total)
            total_data += total
            
            if len(self.buffer) > 500 and has_buffer:
                print(f"[INFO] Emptying buffer of size {len(self.buffer)}")
                self.buffer.empty()
                self.buffer_batches.append(batch)
        
        self._calculate_class_accuracies()
        self.hmms.calculate_class_accuracies()
        self.cnn.calculate_class_accuracies()
        bayes_acc = round(total_accuracy/total_data*100,2)
        hmm_acc = round(self.hmms.total_accuracy/total_data*100,2)
        cnn_acc = round(self.cnn.total_accuracy/total_data*100,2)
        print(f"[ACCURACY] model: {bayes_acc}, hmm: {hmm_acc}, cnn: {cnn_acc}")
    
    def run_novel(self, novel_class, test_data_size=5000, buffer_size=500):
        random.seed(10)
        torch.manual_seed(10)
        generator = torch.Generator()
        generator.manual_seed(10)
        test_data = Subset(self.train_data, [i for i, target in enumerate(self.train_data.targets) if target in novel_class])

        discard = len(test_data) - test_data_size
        test_data_subset, _ = random_split(test_data,
            [test_data_size, discard],
            generator=generator
        )
        print('Testing set has {} instances'.format(len(test_data_subset)))
        
        batch_size = int(test_data_size/20)
        
        test_loader = DataLoader(test_data_subset, batch_size=batch_size, shuffle=True, generator=generator)
        
        for batch, (images, labels) in enumerate(test_loader):
            total = 0
            
            cnn_probabilities = self.cnn.get_probabilities(images)
            hmm_probabilities = self.hmms.get_all_probabilities(images)
            for i in range(len(labels)):

                cnn_probability = cnn_probabilities[i]
                hmm_probability = hmm_probabilities[i]

                cnn_max_probability = torch.max(cnn_probability)
                hmm_probability = torch.nan_to_num(hmm_probability, 0.0)
                hmm_max_probability = torch.max(hmm_probability)
                diff_max_probability = torch.max(abs(cnn_probability-hmm_probability))
                total += 1
                
                if cnn_max_probability >= 0.95 or diff_max_probability <= 0.05:
                    # Normal
                    pred_label = DataType.NORMAL
                elif hmm_max_probability <= 0.3 or (hmm_max_probability > 0.7 and diff_max_probability >= 0.2) or cnn_max_probability < 0.6:
                    # Outlier
                    pred_label = DataType.OUTLIER
                else:
                    # Novelty
                    pred_label = DataType.NOVEL
                    self.buffer.add(images[i])
                
                self.class_accuracies[DataType.NOVEL][pred_label] += 1
                self.class_accuracies[DataType.NOVEL]["total"] += 1
                self.accuracy_over_time[pred_label][batch] += 1
            
            self.accuracy_over_time[DataType.NORMAL][batch] = self._calculate_accuracy(batch, DataType.NORMAL, total)
            self.accuracy_over_time[DataType.NOVEL][batch] = self._calculate_accuracy(batch, DataType.NOVEL, total)
            self.accuracy_over_time[DataType.OUTLIER][batch] = self._calculate_accuracy(batch, DataType.OUTLIER, total)
            if len(self.buffer) > buffer_size:
                print(f"[INFO] Emptying buffer of size {len(self.buffer)}")
                self.buffer.empty()
                self.buffer_batches.append(batch)
        self._calculate_class_accuracies()
        
    def run_novel_converge(self, novel_class, repeats=20, buffer_size=500):
        random.seed(10)
        torch.manual_seed(10)
        generator = torch.Generator()
        generator.manual_seed(10)
        test_data = Subset(self.train_data, [i for i, target in enumerate(self.train_data.targets) if target in novel_class])

        discard = len(test_data) - buffer_size
        test_data_subset, _ = random_split(test_data,
            [buffer_size, discard],
            generator=generator
        )
        print('Testing set has {} instances'.format(len(test_data_subset)))
                
        test_loader = DataLoader(test_data_subset, batch_size=buffer_size, shuffle=True, generator=generator)
        
        for rep in repeats:
            for images, labels in test_loader:
                total = 0
                
                cnn_probabilities = self.cnn.get_probabilities(images)
                hmm_probabilities = self.hmms.get_all_probabilities(images)
                for i in range(len(labels)):

                    cnn_probability = cnn_probabilities[i]
                    hmm_probability = hmm_probabilities[i]

                    cnn_max_probability = torch.max(cnn_probability)
                    hmm_probability = torch.nan_to_num(hmm_probability, 0.0)
                    hmm_max_probability = torch.max(hmm_probability)
                    diff_max_probability = torch.max(abs(cnn_probability-hmm_probability))
                    total += 1
                    
                    if cnn_max_probability >= 0.95 or diff_max_probability <= 0.05:
                        # Normal
                        pred_label = DataType.NORMAL
                    elif hmm_max_probability <= 0.3 or (hmm_max_probability > 0.7 and diff_max_probability >= 0.2) or cnn_max_probability < 0.6:
                        # Outlier
                        pred_label = DataType.OUTLIER
                    else:
                        # Novelty
                        pred_label = DataType.NOVEL
                        self.buffer.add(images[i])
                    
                    self.class_accuracies[DataType.NOVEL][pred_label] += 1
                    self.class_accuracies[DataType.NOVEL]["total"] += 1
                    self.accuracy_over_time[pred_label][rep] += 1
                
                self.accuracy_over_time[DataType.NORMAL][rep] = self._calculate_accuracy(rep, DataType.NORMAL, total)
                self.accuracy_over_time[DataType.NOVEL][rep] = self._calculate_accuracy(rep, DataType.NOVEL, total)
                self.accuracy_over_time[DataType.OUTLIER][rep] = self._calculate_accuracy(rep, DataType.OUTLIER, total)
                if len(self.buffer) > buffer_size:
                    print(f"[INFO] Emptying buffer of size {len(self.buffer)}")
                    self.buffer.empty()
                    self.buffer_batches.append(rep)
            self._calculate_class_accuracies()
    
    def save_accuracy(self, extension, folder):
        if not os.path.exists(f"results_{folder}"):
            os.makedirs(f"results_{folder}")
            
        accuracy_data = pandas.DataFrame.from_dict(self.accuracy_over_time, orient="index")
        accuracy_data.to_csv(f"results_{folder}/accuracy_{extension}.csv")
        
        overall_accuracy = pandas.DataFrame.from_dict(self.class_accuracies)
        overall_accuracy.rename(columns={2: "Actual outlier", 1: "Actual Novel", 0: "Actual Normal"})
        overall_accuracy.to_csv(f"results_{folder}/class_accuracies_{extension}.csv")
        
        #self.hmms.save_accuracy(f"{extension}", f"{folder}")
        #self.cnn.save_accuracy(f"{extension}", f"{folder}")

    def threshold(self, type=DataType.NORMAL, classes=[0,1,2,3,4,5,6,7,8,9], test_data_size=5000, no_thresholds=20):
        random.seed(10)
        torch.manual_seed(10)
        generator = torch.Generator()
        generator.manual_seed(10)
        if type == DataType.NORMAL or type == DataType.NOVEL:
            test_data = self.train_data
        else:
            test_data = self.outlier_data
            
        test_data = Subset(test_data, [i for i, target in enumerate(test_data.targets) if target in classes])

        discard = len(test_data) - test_data_size
        test_data_subset, _ = random_split(test_data,
            [test_data_size, discard],
            generator=generator
        )
        
        test_loader = DataLoader(test_data_subset, batch_size=1000, shuffle=True, generator=generator)
        diff_thresholds = [0] * (no_thresholds + 1)
        hmm_thresholds = [0] * (no_thresholds + 1)
        cnn_thresholds = [0] * (no_thresholds + 1)
        
        for test_images, _ in test_loader:
            cnn_probabilities = self.cnn.get_probabilities(test_images)
            hmm_probabilities = self.hmms.get_all_probabilities(test_images)
            
            for hmm_prob, cnn_prob in zip(hmm_probabilities, cnn_probabilities):
                max_hmm = torch.max(hmm_prob)
                max_cnn = torch.max(cnn_prob)
                if torch.isnan(max_hmm):
                    max_hmm = 0
                max_diff = abs(max_cnn - max_hmm)
                max_diff = int(max_diff * no_thresholds)
                max_hmm = int(max_hmm * no_thresholds)
                max_cnn = int(max_cnn * no_thresholds)
                diff_thresholds[max_diff] += 1
                hmm_thresholds[max_hmm] += 1
                cnn_thresholds[max_cnn] += 1
                
        print('HMM Thresholds:')
        for threshold in hmm_thresholds:
            print('{:.2f}'.format(threshold / test_data_size), end=', ')
        print()
        
        print('CNN Thresholds:')
        for threshold in cnn_thresholds:
            print('{:.2f}'.format(threshold / test_data_size), end=', ')
        print()

        print('Bayes Thresholds:')
        for threshold in diff_thresholds:
            print('{:.2f}'.format(threshold / test_data_size), end=', ')
        print()