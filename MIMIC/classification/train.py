import time
import csv
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import datetime
import torch.optim
import torch.utils.data
from torchvision import  models
from torch import nn
import torch
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from classification.dataset import MIMICCXRDataset
from classification.utils import  checkpoint, save_checkpoint, saved_items
from classification.batchiterator import batch_iterator
from tqdm import tqdm
import random
import numpy as np



def train(train_df, val_df, PATH_TO_IMAGES, modeltype, CRITERION, device,lr):
    """
        This function train the model.
        
        Arguments:
        train_df : train dataframe 
        val_df : validation dataframe 
        PATH_TO_IMAGES: Path to the image directory on the server
        modeltype: It is either densenet for training a densnet model or resume to load the last saved model and resume training
        CRITERION: Loss function to calculate between predictions and outputs. e.g BCE loss
        device: Device on which to run computation
        lr: learning rate
        
        
        Returns:
        The function checlkpoint the best model in the result folder
        model : best trained model
        best_epoch: the epoch number of the best model
       
    """

    # Training parameters
    BATCH_SIZE = 48

    WORKERS = 12  # mean: how many subprocesses to use for data loading.
    N_LABELS = 14
    start_epoch = 0
    num_epochs = 64  # number of epochs to train for (if early stopping is not triggered)

    random_seed = 47 #random.randint(0,100)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        MIMICCXRDataset(train_df, path_image=PATH_TO_IMAGES, transform=transforms.Compose([
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomRotation(15),
                                                                    transforms.Scale(256),
                                                                    transforms.CenterCrop(256),
                                                                    transforms.ToTensor(),
                                                                    normalize
                                                                ])),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        MIMICCXRDataset(val_df,path_image=PATH_TO_IMAGES, transform=transforms.Compose([
                                                                transforms.Scale(256),
                                                                transforms.CenterCrop(256),
                                                                transforms.ToTensor(),
                                                                normalize
                                                            ])),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)

    if modeltype == 'densenet':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features


        model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    
    if modeltype == 'resume':
        CheckPointData = torch.load('./results/checkpoint')
        model = CheckPointData['model']

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        model = nn.DataParallel(model)

    model = model.to(device)
    
    if CRITERION == 'BCELoss':
        criterion = nn.BCELoss().to(device)

    epoch_losses_train = []
    epoch_losses_val = []

    since = time.time()

    best_loss = 999999
    best_epoch = -1
#--------------------------Start of epoch loop
    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
# -------------------------- Start of phase

        phase = 'train'
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        running_loss = batch_iterator(model=model, phase=phase, Data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_train = running_loss / train_df_size
        epoch_losses_train.append(epoch_loss_train.item())
        print("Train_losses:", epoch_losses_train)

        phase = 'val'
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        running_loss = batch_iterator(model=model, phase=phase, Data_loader=val_loader, criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_val = running_loss / val_df_size
        epoch_losses_val.append(epoch_loss_val.item())
        print("Validation_losses:", epoch_losses_val)

        # checkpoint model if has best val loss yet
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            checkpoint(model, best_loss, best_epoch, lr)

                # log training and validation loss over each epoch
        with open("results/log_train", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if (epoch == 1):
                logwriter.writerow(["epoch", "train_loss", "val_loss","seed","lr"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val,random_seed, lr])
# -------------------------- End of phase

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            if epoch_loss_val > best_loss:
                print("decay loss from " + str(lr) + " to " + str(lr / 2) + " as not seeing improvement in val loss")
                lr = lr / 2
                print("created new optimizer with lr " + str(lr))
                if ((epoch - best_epoch) >= 10):
                    print("no improvement in 10 epochs, break")
                    break
        #old_epoch = epoch 
    #------------------------- End of epoch loop
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    saved_items(epoch_losses_train, epoch_losses_val, time_elapsed, BATCH_SIZE)
    #
    checkpoint_best = torch.load('./results/checkpoint')
    model = checkpoint_best['model']

    best_epoch = checkpoint_best['best_epoch']
    print(best_epoch)



    return model, best_epoch


