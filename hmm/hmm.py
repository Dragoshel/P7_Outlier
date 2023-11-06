from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM

from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST, MNIST
import torch

import numpy

print("[INFO] loading the KMNIST test dataset...")
train_data = KMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = KMNIST(root='data', train=False, download=True, transform=ToTensor())

model = DenseHMM([Normal(), Normal(), Normal(), Normal(), Normal(), Normal(), Normal(), Normal(), Normal(), Normal()], max_iter=10, verbose=True)

print("[INFO] initializing hmm...")
model._initialize(torch.randn(60000, 784, 1))
#model.fit(torch.randn(60000, 784, 1))

print("[INFO] processing images...")
flattened_images = []
for data, target in train_data:
    image = data[0]
    flattened_image = image.view(-1, 1)
    flattened_images.append(flattened_image)


flattened_images = numpy.array(flattened_images)
flattened_images = flattened_images[:1000]

print("[INFO] fitting HMM...")
model.fit(flattened_images)


# from pomegranate.distributions import Categorical
# from pomegranate.hmm import DenseHMM

# import numpy

# d1 = Categorical([[0.25, 0.25, 0.25, 0.25]])
# d2 = Categorical([[0.10, 0.40, 0.40, 0.10]])

# model = DenseHMM()
# model.add_distributions([d1, d2])

# model.add_edge(model.start, d1, 0.5)
# model.add_edge(model.start, d2, 0.5)
# model.add_edge(d1, d1, 0.9)
# model.add_edge(d1, d2, 0.1)
# model.add_edge(d2, d1, 0.1)
# model.add_edge(d2, d2, 0.9)

# X3 = torch.randn(1, 51, 1)
# model.fit(X3)

# sequence = 'CGACTACTGACTACTCGCCGACGCGACTGCCGTCTATACTGCGCATACGGC'
# X = numpy.array([[[['A', 'C', 'G', 'T'].index(char)] for char in sequence]])
# X.shape

# #model.fit(X)
# y_hat = model.predict(X)

# print("sequence: {}".format(''.join(sequence)))
# print("hmm pred: {}".format(''.join([str(y.item()) for y in y_hat[0]])))


# from pomegranate.markov_chain import MarkovChain

# model = MarkovChain(k=10)

# print("[INFO] processing images...")
# flattened_images = []
# for data, target in train_data:
#     image = data[0]
#     flattened_image = image.view(-1, 1)
#     flattened_images.append(flattened_image)

# flattened_images = numpy.array(flattened_images)

# model.fit(flattened_images)



