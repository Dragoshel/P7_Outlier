import time

import torch
from cnn.cnn import CNN
from utils.classes import index_labels

def train_model(epochs: int, training_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader, model: CNN, optimizer, criterion):
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
    for i, (images, labels) in enumerate(training_loader):
        print(f"[INFO] Running batch {i} of {len(training_loader)}")
        images = images.to(torch.float32)
        labels = torch.tensor(index_labels(labels.tolist()))
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def _validation_pass(validation_loader: torch.utils.data.DataLoader, model: CNN, criterion) -> float:
    for images, labels in validation_loader:
        images = images.to(torch.float32)
        labels = torch.tensor(index_labels(labels.tolist()))
        outputs = model(images)
        validity = criterion(outputs, labels)
    
    return validity.item()
