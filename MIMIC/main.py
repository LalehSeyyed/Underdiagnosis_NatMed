import torch
from classification.train import train
from classification.prediction import make_pred_multilabel
import pandas as pd

#----------------------------- q
PATH_TO_IMAGES = "/PATH TO DATASET IMAGES IN YOUR SERVER/MIMIC-CXR/"

TRAIN_DF_PATH = "/PATH TO DATASET CSV FILES IN YOUR SERVER/split/new_train.csv"
TEST_DF_PATH  = "/PATH TO DATASET CSV FILES IN YOUR SERVER/split/new_test.csv"
VAL_DF_PATH   = "/PATH TO DATASET CSV FILES IN YOUR SERVER/split/new_valid.csv"

# We mix all existing data of the provoder regardless of their original validation/train label in the original dataset and split them into 80-10-10 train test and validation sets based on Patient-ID such that no patient images appears in more than one split. 


def main():

    MODE = "train"  # Select "train" or "test", "Resume"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_df = pd.read_csv(VAL_DF_PATH)
    val_df_size = len(val_df)
    print("Validation_df size:",val_df_size)

    train_df = pd.read_csv(TRAIN_DF_PATH)
    train_df_size = len(train_df)
    print("Train_df size:", train_df_size)
    
    test_df = pd.read_csv(TEST_DF_PATH)
    test_df_size = len(test_df)
    print("Test_df size:", test_df_size)


    if MODE == "train":
        ModelType = "densenet"  # currently code is based on densenet121 
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = train(train_df, val_df, PATH_TO_IMAGES, ModelType, CriterionType, device,LR)

        


    if MODE =="test":
        val_df = pd.read_csv(VAL_DF_PATH)
        test_df = pd.read_csv(TEST_DF_PATH)

        CheckPointData = torch.load('./classification/results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, PATH_TO_IMAGES, device)


    if MODE == "Resume":
        ModelType = "Resume" 
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = train(train_df, val_df, PATH_TO_IMAGES, ModelType, CriterionType, device,LR)
        
        
        PlotLearnignCurve()


if __name__ == "__main__":
    main()
