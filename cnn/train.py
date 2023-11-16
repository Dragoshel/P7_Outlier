import time

import torch
from cnn.cnn import CNN
from utils.data_types import DataType
from utils.images import show_images
from utils.classes import index_labels

def train_model(epochs: int, training_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader, model: CNN, optimizer, criterion) -> tuple:
    """Perform the training of the model, first running one training batch and then validating the models accuracy,
    during the validation pass gradient descent is turned off, so the weights are not adjusted.

    Args:
        epochs (int): Number of times the model should be trained
        training_loader (DataLoader): Contains the dataset for which to train the model
        validation_loader (DataLoader): Contains the dataset for which to validate the model
        model (CNN): Model to train
        optimizer: Optimisation function be used on the backward pass
        criterion: Criterion to evaluate the loss of the model with, to be used on the forward pass

    Returns:
        tuple: Lists with the training and validation loss experienced in each batch
    """    
    
    train_loss = []
    valid_loss = []
    
    starttime = time.time()
    lasttime = starttime
    
    for epoch in range(epochs):
        loss = _training_pass(training_loader, model, optimizer, criterion)
        train_loss.append(loss)
        with torch.no_grad():
            validity = _validation_pass(validation_loader, model, criterion)
            valid_loss.append(validity)
        laptime = round((time.time() - lasttime), 2)
        totaltime = round((time.time() - starttime), 2)
        lasttime = time.time()
        print('Epoch [{}/{}], Loss: {:.4f}, Validity: {:.4}, Lap time: {}, Total time: {}'.format(epoch+1, epochs, loss, validity, laptime, totaltime))
    
    return train_loss, valid_loss

def _training_pass(training_loader: torch.utils.data.DataLoader, model: CNN, optimizer, criterion) -> float:
    """Train the model with one batch at a time, calculating the loss during the forward pass
    and then adjusting the weights on the backward pass.

    Args:
        training_loader (DataLoader): Loader containg the batches training data
        model (CNN): Model to train
        optimizer: Optimisation function be used on the backward pass
        criterion: Criterion to evaluate the loss of the model with, to be used on the forward pass

    Returns:
        float: Loss of the given training pass
    """
    for i, (images, labels) in enumerate(training_loader):
        if i == 0:
            show_images(images, True)
        labels = torch.tensor(index_labels(labels.tolist()))
        images = images
        labels = labels
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def _validation_pass(validation_loader: torch.utils.data.DataLoader, model: CNN, criterion) -> float:
    """Validate the model with one batch at a time, calculating the loss at each forward pass

    Args:
        validation_loader (torch.utils.data.DataLoader): Loader containg the batches for the validation data
        model (CNN): Model to validate
        criterion: Criterion to evaluate the loss of the model with, to be used on the forward pass

    Returns:
        float: Loss of the given validation pass
    """    
    for i, (images, labels) in enumerate(validation_loader):
        if i == 0:
            show_images(images, True)
        labels = torch.tensor(index_labels(labels.tolist()))
        images = images
        labels = labels
        outputs = model(images)
        validity = criterion(outputs, labels)
    
    return validity.item()
