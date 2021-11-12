import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.misc import imread, imresize
from PIL import Image




class MIMICCXRDataset(Dataset):
    def __init__(self, dataframe, path_image, finding="any", transform=None):
        self.dataframe = dataframe
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.finding = finding
        self.transform = transform
        self.path_image = path_image

        self.PRED_LABEL = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Lesion',
            'Airspace Opacity',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]

        img = imread(os.path.join(self.path_image, item["path"]))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)
        # img = imresize(img, (256, 256))
        # img = img.transpose(2, 0, 1)
        # assert img.shape == (3, 256, 256)
        # assert np.max(img) <= 255
        # img = torch.FloatTensor(img / 255.)
        if self.transform is not None:
            img = self.transform(img)

        # label = np.zeros(len(self.PRED_LABEL), dtype=int)
        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        for i in range(0, len(self.PRED_LABEL)):

            if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')

        return img, label, item["path"]#self.dataframe.index[idx]

    def __len__(self):
        return self.dataset_size