import numpy as np
import torch
import torchvision
from PIL import Image

class DatasetFromNpy(torch.utils.data.Dataset):

    def __init__(self, data_npy, labels_npy, transform=None, target_transform=None):
        self.data = np.load(data_npy)
        self.targets = np.load(labels_npy)
        self.transform = transform
        self.target_transform = target_transform

        if len(self.data.shape) < 4:
            self.grayscale = True
        else:
            self.grayscale = False

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.grayscale:
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return self.data.shape[0]


"""Example of loading the BadLabels Dataset."""
badset = DatasetFromNpy('training_data.npy',
                        'training_noisy_labels.npy',
                        torchvision.transforms.ToTensor())
