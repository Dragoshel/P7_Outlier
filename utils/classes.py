from utils.data_types import DataType
import random
import sys

this = sys.modules[__name__]

this.normal_classes = []
this.novel_classes = []
this.no_normal = 0

def pick_classes(no_normal: int):
    mnist_classes = range(0,10)
    this.no_normal = no_normal
    this.normal_classes = random.sample(mnist_classes, no_normal)
    this.novel_classes = [no for no in mnist_classes if no not in this.normal_classes]
    print(f'Normal classes: {this.normal_classes}, Novel classes: {this.novel_classes}')

def index_labels(labels: list) -> list:
    idx_labels = []
    for label in labels:
        if label in this.normal_classes:
            idx = this.normal_classes.index(label)
        elif label in this.novel_classes:
            idx = this.novel_classes.index(label) + 1 + this.no_normal
        else:
            idx = label
        idx_labels.append(idx)
    return idx_labels

def get_normal_classes() -> list:
    return this.normal_classes

def get_novel_classes() -> list:
    return this.novel_classes
