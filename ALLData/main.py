import torch
from classification.train import train
from classification.prediction import make_pred_multilabel
import pandas as pd
from Config import train_df, test_df, val_df


def main():

    MODE = "train"  # Select "train" or "test", "resume"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if MODE == "train":
        modeltype = "densenet"  
        CRITERION = 'BCELoss'
        lr = 0.5e-3

        model, best_epoch = train( modeltype, CRITERION, device,lr)


    if MODE =="test":
       
        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, device)


    if MODE == "resume":
        modeltype = "resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CRITERION = 'BCELoss'
        lr = 0.5e-3

        model, best_epoch = train( modeltype, CRITERION, device,lr)

      


    


if __name__ == "__main__":
    main()
