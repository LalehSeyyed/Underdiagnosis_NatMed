import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.misc import imread
from PIL import Image


class NIH(Dataset):
    def __init__(self, dataframe, PATH_TO_IMAGES, transform=None):
        
        """
            Dataset class representing ChestX-ray14 dataset
            
            Arguments:
            dataframe: Whether the dataset represents the train, test, or validation split
            PATH_TO_IMAGES: Path to the image directory on the server
            transform: Whether conduct transform to the images or not
            
            Returns:
            image, label and item["Image Index"] as the unique indicator of each item in the dataloader.
        """
        
        
        self.dataframe = dataframe
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.PATH_TO_IMAGES = PATH_TO_IMAGES

        self.PRED_LABEL = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'No Finding',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]

        img = imread(os.path.join(self.PATH_TO_IMAGES, item["Image Index"]))
        
        
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        if len(img.shape)>2:
            img = img[:,:,0]
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        
        for i in range(0, len(self.PRED_LABEL)):
            if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')
            
        return img, label, item["Image Index"]

    def __len__(self):
        return self.dataset_size

