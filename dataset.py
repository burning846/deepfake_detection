import torch
import torch.utils
import torch.utils.data

import os
import pandas as pd
import cv2

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_root, transform=None, is_train=False):
        super(TrainingDataset, self).__init__()

        self.df = pd.read_csv(df_path)
        self.img_root = img_root
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.img_root, self.df['img_name'][index]))
        label = int(self.df['target'][index])

        if self.transform:
            img = self.transform(img)


        return img, label
    

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, img_root, transform=None, is_train=False):
        super(InferenceDataset, self).__init__()

        self.df = pd.read_csv(df_path)
        self.img_root = img_root
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.img_root, self.df['img_name'][index]))

        if self.transform:
            img = self.transform(img)

        return img