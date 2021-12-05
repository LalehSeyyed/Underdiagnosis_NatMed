import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.misc import imread, imresize
from PIL import Image



class AllDatasetsShared(Dataset):
    def __init__(self, dataframe, finding="any", transform=None):
        
                
        """
            Dataset class representing the aggregation on all three datasets.
            Initially in the Config.py we have aggregated all CheCpert, MIMIC-CXR and ChestX-ray14 datasets on the 8 shared labels. 
            
            Arguments:
            dataframe: Whether the dataset represents the train, test, or validation split
            PATH_TO_IMAGES: Path to the image directory on the server
            transform: Whether conduct transform to the images or not
            
            Returns:
            image, label and item["Jointpath"] as the unique indicator of each item in the dataloader.
        """
        
        
        
        self.dataframe = dataframe
        self.dataset_size = self.dataframe.shape[0]        
        self.transform = transform      
        self.PRED_LABEL = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]

        img = imread(item["Jointpath"])
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

        return img, label, item["Jointpath"]

    def __len__(self):
        return self.dataset_size
