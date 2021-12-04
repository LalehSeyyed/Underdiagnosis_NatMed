
import torch
from classification.utils import clip_gradient
import numpy as np

def batch_iterator(model, 
                   phase,
                   dataloader,
                   criterion,
                   optimizer,
                   device):
    
    '''
        It is a function that iterates over the data in dataloader.
        It evaluate (in val/test phase) or train (in train phase) the model and returns the loss.
        
        Arguments:
        model : Base model
        phase : A string to that denotes if it is "train" or 'val'
        dataloader : The associated data loader
        criterion : Loss function to calculate between predictions and outputs.
        optimizer : Optimiser to calculate gradient step from loss
        device: Device on which to run computation
        
    '''

    
    grad_clip = 0.5 

    print_freq = 1000
    running_loss = 0.0

    
    for i, data in enumerate(dataloader):
        imgs, labels, _ = data
        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        if phase == "train":
            optimizer.zero_grad()
            model.train()
            outputs = model(imgs)
        else:

            model.eval()
            with torch.no_grad():
                outputs = model(imgs)


        loss = criterion(outputs, labels)

        if phase == 'train':

            loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()  # update weights

        running_loss += loss * batch_size
        if (i % 200 == 0):
            print(str(i * batch_size))




    return running_loss
