from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM

from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST, MNIST
import torch

import numpy

print("[INFO] loading the KMNIST test dataset...")
train_data = KMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = KMNIST(root='data', train=False, download=True, transform=ToTensor())

states = [Normal(), Normal(), Normal(), Normal(), Normal()]
model = DenseHMM(states, verbose=True)

print("[INFO] processing images...")
flattened_images = []
for data, target in train_data:
	image = data[0]
	flattened_image = image.view(-1, 1)		
	flattened_images.append(flattened_image)

flattened_images = numpy.array(flattened_images)

# Something quick
flattened_images = numpy.array([flattened_images[0],flattened_images[1], flattened_images[2]])
print(flattened_images.shape)

print("[INFO] fitting HMM...")
model.fit(flattened_images)
