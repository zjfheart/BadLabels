"""
Original Code is from https://github.com/shengliu66/SOP
"""

import json
import sys
import os

import pandas as pd
import torch
import torchvision
from PIL import Image
from typing import Tuple, Union, Optional

import numpy as np
from numpy.testing import assert_array_almost_equal
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms
sys.path.append("../..")
from generic.sop_parse_config import ConfigParser
from augment.augmentations import Augmentation, CutoutDefault
from augment.augmentation_archive import autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10#, autoaug_imagenet_policy, svhn_policies


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    valid_sampler: Optional[SubsetRandomSampler]
    sampler: Optional[SubsetRandomSampler]

    def __init__(self, train_dataset, batch_size, shuffle, validation_split: float, num_workers, pin_memory,
                 collate_fn=default_collate, val_dataset=None):
        self.collate_fn = collate_fn
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.val_dataset = val_dataset

        self.batch_idx = 0
        self.n_samples = len(train_dataset) if val_dataset is None else len(train_dataset) + len(val_dataset)
        self.init_kwargs = {
            'dataset': train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }
        if val_dataset is None:
            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
            super().__init__(sampler=self.sampler, **self.init_kwargs)
        else:
            super().__init__(**self.init_kwargs)

    def _split_sampler(self, split) -> Union[Tuple[None, None], Tuple[SubsetRandomSampler, SubsetRandomSampler]]:
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        print(f"Train: {len(train_sampler)} Val: {len(valid_sampler)}")

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self, bs = 1000):
        if self.val_dataset is not None:
            kwargs = {
                'dataset': self.val_dataset,
                'batch_size': bs,
                'shuffle': False,
                'collate_fn': self.collate_fn,
                'num_workers': self.num_workers
            }
            return DataLoader(**kwargs)
        else:
            print('Using sampler to split!')
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

def get_cifar10(root, cfg_trainer, train=True,
                transform_train=None, transform_train_aug=None, transform_val=None,
                download=False, noise_file=''):
    base_dataset = torchvision.datasets.CIFAR10(root, train=train, download=download)
    if train:
        train_idxs, val_idxs = c10_train_val_split(base_dataset.targets)
        train_dataset = CIFAR10_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train,
                                      transform_aug=transform_train_aug)
        val_dataset = CIFAR10_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)

        if cfg_trainer['asym']:
            train_dataset.asymmetric_noise(noise_file)
            val_dataset.asymmetric_noise(noise_file)
        elif cfg_trainer['instance']:
            train_dataset.instance_noise(noise_file)
            val_dataset.instance_noise(noise_file)
        elif cfg_trainer['att']:
            train_dataset.contamination_noise(noise_file)
            val_dataset.contamination_noise(noise_file)
        else:
            train_dataset.symmetric_noise(noise_file)
            val_dataset.symmetric_noise(noise_file)

        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        train_dataset = []
        val_dataset = CIFAR10_val(root, cfg_trainer, None, train=train, transform=transform_val)
        print(f"Test: {len(val_dataset)}")

    return train_dataset, val_dataset

def c10_train_val_split(base_dataset: torchvision.datasets.CIFAR10):
    num_classes = 10
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs

class CIFAR10_train(torchvision.datasets.CIFAR10):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, transform_aug=None, target_transform=None,
                 download=False):
        super(CIFAR10_train, self).__init__(root, train=train,
                                            transform=transform,
                                            target_transform=target_transform,
                                            download=download)
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data[indexs]  # self.train_data[indexs]
        self.train_labels = np.array(self.targets)[indexs]  # np.array(self.train_labels)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        self.transform_aug = transform_aug

        self.train_labels_gt = self.train_labels.copy()

    def symmetric_noise(self, noise_file):
        # self.train_labels_gt = self.train_labels.copy()
        # np.random.seed(seed=888)
        if noise_file is not None:
            noise_label = json.load(open(noise_file, "r"))
            noisylabel = np.array(noise_label)[self.indexs]
            self.train_labels = np.array(noisylabel)
        else:
            indices = np.random.permutation(len(self.train_data))
            for i, idx in enumerate(indices):
                if i < self.cfg_trainer['percent'] * len(self.train_data):
                    self.noise_indx.append(idx)
                    noiselabels = list(range(self.num_classes))
                    noiselabels.remove(self.train_labels[idx])
                    self.train_labels[idx] = np.random.choice(np.array(noiselabels), size=1)
                    # self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)


    def asymmetric_noise(self, noise_file):
        # self.train_labels_gt = self.train_labels.copy()
        if noise_file is not None:
            noise_label = json.load(open(noise_file, "r"))
            noisylabel = np.array(noise_label)[self.indexs]
            self.train_labels = np.array(noisylabel)
        else:
            for i in range(self.num_classes):
                indices = np.where(self.train_labels == i)[0]
                np.random.shuffle(indices)
                for j, idx in enumerate(indices):
                    if j < self.cfg_trainer['percent'] * len(indices):
                        self.noise_indx.append(idx)
                        # truck -> automobile
                        if i == 9:
                            self.train_labels[idx] = 1
                        # bird -> airplane
                        elif i == 2:
                            self.train_labels[idx] = 0
                        # cat -> dog
                        elif i == 3:
                            self.train_labels[idx] = 5
                        # dog -> cat
                        elif i == 5:
                            self.train_labels[idx] = 3
                        # deer -> horse
                        elif i == 4:
                            self.train_labels[idx] = 7


    def instance_noise(self, noise_file):

        noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))

        noisylabel = np.array(noise_label)[self.indexs]
        # truelabel = noise_label['clean_label_train'][self.indexs]

        self.train_labels = np.array(noisylabel)

    def contamination_noise(self, noise_file):

        noise_label = json.load(open(noise_file,"r"))
        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))

        noisylabel = np.array(noise_label)[self.indexs]
        # truelabel = noise_label['clean_label_train'][self.indexs]

        self.train_labels = np.array(noisylabel)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform_aug is not None:
            img2 = self.transform_aug(img)
        else:
            img2 = self.transform(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img2, target, index, target_gt

    def __len__(self):
        return len(self.train_data)

class CIFAR10_val(torchvision.datasets.CIFAR10):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        # self.train_data = self.data[indexs]
        # self.train_labels = np.array(self.targets)[indexs]
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        if train:
            self.train_data = self.data[indexs]
            self.train_labels = np.array(self.targets)[indexs]
        else:
            self.train_data = self.data
            self.train_labels = np.array(self.targets)
        self.train_labels_gt = self.train_labels.copy()
        self.indexs = indexs

    def symmetric_noise(self, noise_file):
        if noise_file is not None:
            noise_label = json.load(open(noise_file, "r"))
            noisylabel = np.array(noise_label)[self.indexs]
            self.train_labels = np.array(noisylabel)
        else:
            indices = np.random.permutation(len(self.train_data))
            for i, idx in enumerate(indices):
                if i < self.cfg_trainer['percent'] * len(self.train_data):
                    self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)


    def asymmetric_noise(self, noise_file):
        if noise_file is not None:
            noise_label = json.load(open(noise_file, "r"))
            noisylabel = np.array(noise_label)[self.indexs]
            self.train_labels = np.array(noisylabel)
        else:
            for i in range(self.num_classes):
                indices = np.where(self.train_labels == i)[0]
                np.random.shuffle(indices)
                for j, idx in enumerate(indices):
                    if j < self.cfg_trainer['percent'] * len(indices):
                        # truck -> automobile
                        if i == 9:
                            self.train_labels[idx] = 1
                        # bird -> airplane
                        elif i == 2:
                            self.train_labels[idx] = 0
                        # cat -> dog
                        elif i == 3:
                            self.train_labels[idx] = 5
                        # dog -> cat
                        elif i == 5:
                            self.train_labels[idx] = 3
                        # deer -> horse
                        elif i == 4:
                            self.train_labels[idx] = 7


    def instance_noise(self, noise_file):

        self.train_labels_gt = self.train_labels.copy()

        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))
        noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))

        noisylabel = np.array(noise_label)[self.indexs]

        self.train_labels = np.array(noisylabel)

    def contamination_noise(self, noise_file):
        self.train_labels_gt = self.train_labels.copy()

        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))
        noise_label = json.load(open(noise_file,"r"))

        noisylabel = np.array(noise_label)[self.indexs]

        self.train_labels = np.array(noisylabel)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt


def get_cifar100(root, cfg_trainer, train=True,
                 transform_train=None, transform_train_aug=None, transform_val=None,
                 download=False, noise_file=''):
    base_dataset = torchvision.datasets.CIFAR100(root, train=train, download=download)
    if train:
        train_idxs, val_idxs = c100_train_val_split(base_dataset.targets)
        train_dataset = CIFAR100_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train,
                                       transform_aug=transform_train_aug)
        val_dataset = CIFAR100_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)
        if cfg_trainer['asym']:
            train_dataset.asymmetric_noise()
            val_dataset.asymmetric_noise()
        elif cfg_trainer['instance']:
            train_dataset.instance_noise(noise_file)
            val_dataset.instance_noise(noise_file)
        elif cfg_trainer['att']:
            train_dataset.contamination_noise(noise_file)
            val_dataset.contamination_noise(noise_file)
        # elif cfg_trainer['real'] == 'noisy_label':
        #     new_targets = cifar100n(root, 'noisy_label', download)
        #     train_dataset.train_labels = np.array(new_targets)[train_dataset.indexs]
        else:
            train_dataset.symmetric_noise()
            val_dataset.symmetric_noise()

        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        train_dataset = []
        val_dataset = CIFAR100_val(root, cfg_trainer, None, train=train, transform=transform_val)
        print(f"Test: {len(val_dataset)}")

    return train_dataset, val_dataset

# def download_cifarn(root):
#     wget.download('http://128.114.59.66:5000/files/CIFAR-N.zip', out=root)
#     with ZipFile(os.path.join(root, 'CIFAR-N.zip'), 'r') as f:
#         f.extractall(root)
#
# def cifar100n(root, key, download=True):
#     label_path = os.path.join(root, 'CIFAR-N', 'CIFAR-100_human.pt')
#     if not os.path.exists(label_path):
#         if download:
#             download_cifarn(root)
#         else:
#             raise FileNotFoundError(f'Labels not found. Please set download=True. Path: {data_path}')
#     noise_label = torch.load(label_path)
#     targets_new = torch.from_numpy(noise_label[key])
#     return targets_new

def c100_train_val_split(base_dataset: torchvision.datasets.CIFAR100):
    num_classes = 100
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs

class CIFAR100_train(torchvision.datasets.CIFAR100):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, transform_aug=None,
                 target_transform=None,
                 download=False):
        super(CIFAR100_train, self).__init__(root, train=train,
                                             transform=transform,
                                             target_transform=target_transform,
                                             download=download)
        self.num_classes = 100
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data[indexs]
        self.train_labels = np.array(self.targets)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        self.transform_aug = transform_aug
        # self.all_refs_encoded = torch.zeros(self.num_classes,self.num_ref,1024, dtype=np.float32)

        self.count = 0

        self.train_labels_gt = self.train_labels.copy()

    def symmetric_noise(self):

        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                self.noise_indx.append(idx)
                noiselabels = list(range(self.num_classes))
                noiselabels.remove(self.train_labels[idx])
                self.train_labels[idx] = np.random.choice(np.array(noiselabels), size=1)
                # self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)
                # self.noise_indx.append(idx)
                # self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    def build_for_cifar100(self, size, noise):
        """ random flip between two random classes.
        """
        assert (noise >= 0.) and (noise <= 1.)

        P = np.eye(size)
        cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
        P[cls1, cls2] = noise
        P[cls2, cls1] = noise
        P[cls1, cls1] = 1.0 - noise
        P[cls2, cls2] = 1.0 - noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P

    def asymmetric_noise(self, asym=False, random_shuffle=False):
        P = np.eye(self.num_classes)
        n = self.cfg_trainer['percent']
        nb_superclasses = 20
        nb_subclasses = 5

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i + 1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(self.train_labels, P=P,
                                                    random_state=0)
            actual_noise = (y_train_noisy != self.train_labels).mean()
            assert actual_noise > 0.0
            self.train_labels = y_train_noisy

    def instance_noise(self, noise_file):

        noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))

        noisylabel = np.array(noise_label)[self.indexs]
        # truelabel = noise_label['clean_label_train'][self.indexs]

        self.train_labels = np.array(noisylabel)

    def contamination_noise(self, noise_file):

        noise_label = json.load(open(noise_file,"r"))
        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))

        noisylabel = np.array(noise_label)[self.indexs]
        # truelabel = noise_label['clean_label_train'][self.indexs]

        self.train_labels = np.array(noisylabel)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform_aug is not None:
            img2 = self.transform_aug(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img2, target, index, target_gt

    def __len__(self):
        return len(self.train_data)

class CIFAR100_val(torchvision.datasets.CIFAR100):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_val, self).__init__(root, train=train,
                                           transform=transform, target_transform=target_transform,
                                           download=download)

        # self.train_data = self.data[indexs]
        # self.train_labels = np.array(self.targets)[indexs]
        self.num_classes = 100
        self.cfg_trainer = cfg_trainer
        if train:
            self.train_data = self.data[indexs]
            self.train_labels = np.array(self.targets)[indexs]
        else:
            self.train_data = self.data
            self.train_labels = np.array(self.targets)
        self.train_labels_gt = self.train_labels.copy()
        self.indexs = indexs

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.cfg_trainer['percent'] * len(self.train_data):
                noiselabels = list(range(self.num_classes))
                noiselabels.remove(self.train_labels[idx])
                self.train_labels[idx] = np.random.choice(np.array(noiselabels), size=1)
                # self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    def build_for_cifar100(self, size, noise):
        """ random flip between two random classes.
        """
        assert (noise >= 0.) and (noise <= 1.)

        P = np.eye(size)
        cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
        P[cls1, cls2] = noise
        P[cls2, cls1] = noise
        P[cls1, cls1] = 1.0 - noise
        P[cls2, cls2] = 1.0 - noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P

    def asymmetric_noise(self, asym=False, random_shuffle=False):
        P = np.eye(self.num_classes)
        n = self.cfg_trainer['percent']
        nb_superclasses = 20
        nb_subclasses = 5

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i + 1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(self.train_labels, P=P,
                                                    random_state=0)
            actual_noise = (y_train_noisy != self.train_labels).mean()
            assert actual_noise > 0.0
            self.train_labels = y_train_noisy

    def instance_noise(self, noise_file):

        self.train_labels_gt = self.train_labels.copy()

        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))
        noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))

        noisylabel = np.array(noise_label)[self.indexs]

        self.train_labels = np.array(noisylabel)

    def contamination_noise(self, noise_file):
        self.train_labels_gt = self.train_labels.copy()

        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))
        noise_label = json.load(open(noise_file,"r"))

        noisylabel = np.array(noise_label)[self.indexs]

        self.train_labels = np.array(noisylabel)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt


class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0,  training=True, num_workers=4,  pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if config['train_loss']['args']['ratio_consistency'] > 0:
            transform_train_aug = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            if config['data_augmentation']['type'] is not None:

                autoaug = transforms.Compose([])
                # if isinstance(cfg_trainer['aug'], list):
                #     autoaug.transforms.insert(0, Augmentation(C.get()['aug']))
                # else:
                if config['data_augmentation']['type'] == 'fa_reduced_cifar10':
                    autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
                elif config['data_augmentation']['type'] == 'autoaug_cifar10':
                    autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
                elif config['data_augmentation']['type'] == 'autoaug_extend':
                    autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
                elif config['data_augmentation']['type'] == 'default':
                    pass
                else:
                    raise ValueError('not found augmentations. %s' % config['data_augmentation']['type'])
                transform_train_aug.transforms.insert(0, autoaug)

                if config['data_augmentation']['cutout'] > 0:
                    transform_train_aug.transforms.append(CutoutDefault(config['data_augmentation']['cutout']))
        else:
            transform_train_aug = None


        self.data_dir = data_dir

        noise_file=cfg_trainer['noise_file']
        
        self.train_dataset, self.val_dataset = get_cifar10(config['data_loader']['args']['data_dir'], cfg_trainer, train=training,
                                                           transform_train=transform_train, transform_train_aug=transform_train_aug, 
                                                           transform_val=transform_val, noise_file = noise_file)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
    # def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
    #     super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
    #                      val_dataset = self.val_dataset)

    def run_loader(self):
        pass

class CIFAR100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True,num_workers=4, pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']
        
        transform_train = transforms.Compose([
                #transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_train_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if config['data_augmentation']['type'] is not None:

            autoaug = transforms.Compose([])
            # if isinstance(cfg_trainer['aug'], list):
            #     autoaug.transforms.insert(0, Augmentation(C.get()['aug']))
            # else:
            if config['data_augmentation']['type'] == 'fa_reduced_cifar10':
                autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
            elif config['data_augmentation']['type'] == 'autoaug_cifar10':
                autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
            elif config['data_augmentation']['type'] == 'autoaug_extend':
                autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
            elif config['data_augmentation']['type'] == 'default':
                pass
            else:
                raise ValueError('not found augmentations. %s' % config['data_augmentation']['type'])
            transform_train_aug.transforms.insert(0, autoaug)
            # transform_train.transforms.insert(0, autoaug)

            if config['data_augmentation']['cutout'] > 0:
                transform_train_aug.transforms.append(CutoutDefault(config['data_augmentation']['cutout']))

        self.data_dir = data_dir
        config = ConfigParser.get_instance()

        noise_file=cfg_trainer['noise_file']

        self.train_dataset, self.val_dataset = get_cifar100(config['data_loader']['args']['data_dir'], cfg_trainer, train=training,
                                                           transform_train=transform_train, transform_train_aug=transform_train_aug, 
                                                           transform_val=transform_val, noise_file = noise_file)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)
    # def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
    #     super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
    #                      val_dataset = self.val_dataset)

    def run_loader(self):
        pass


class MNIST(torchvision.datasets.MNIST):
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'MNIST', 'processed')

def get_mnist(root, cfg_trainer, train=True,
                transform_train=None, transform_train_aug=None, transform_val=None,
                download=False, noise_file=''):
    base_dataset = torchvision.datasets.MNIST(root, train=train, download=download)
    if train:
        train_idxs, val_idxs = mnist_train_val_split(base_dataset.targets)
        train_dataset = mnist_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train,
                                      transform_aug=transform_train_aug, download=download)
        val_dataset = mnist_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val, download=download)

        if cfg_trainer['instance']:
            train_dataset.instance_noise(noise_file)
            val_dataset.instance_noise(noise_file)
        elif cfg_trainer['att']:
            train_dataset.contamination_noise(noise_file)
            val_dataset.contamination_noise(noise_file)
        else:
            train_dataset.symmetric_noise(noise_file)
            val_dataset.symmetric_noise(noise_file)

        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        train_dataset = []
        val_dataset = mnist_val(root, cfg_trainer, None, train=train, transform=transform_val)
        print(f"Test: {len(val_dataset)}")

    return train_dataset, val_dataset

def mnist_train_val_split(base_dataset: torchvision.datasets.MNIST):
    num_classes = 10
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs

class mnist_train(MNIST):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, transform_aug=None, target_transform=None,
                 download=False):
        super(mnist_train, self).__init__(root, train=train,
                                            transform=transform,
                                            target_transform=target_transform,
                                            download=download)
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        self.train_imgs = self.data[indexs]  # self.train_data[indexs]
        self.train_targets = np.array(self.targets)[indexs]  # np.array(self.train_labels)[indexs]
        self.indexs = indexs
        self.prediction = np.zeros((len(self.train_imgs), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        self.transform_aug = transform_aug

        self.train_targets_gt = self.train_targets.copy()

    def symmetric_noise(self, noise_file):
        # self.train_labels_gt = self.train_labels.copy()
        # np.random.seed(seed=888)
        if noise_file is not None:
            noise_label = json.load(open(noise_file, "r"))
            noisylabel = np.array(noise_label)[self.indexs]
            self.train_targets = np.array(noisylabel)
        else:
            indices = np.random.permutation(len(self.train_imgs))
            for i, idx in enumerate(indices):
                if i < self.cfg_trainer['percent'] * len(self.train_imgs):
                    self.noise_indx.append(idx)
                    noiselabels = list(range(self.num_classes))
                    noiselabels.remove(self.train_targets[idx])
                    self.train_targets[idx] = np.random.choice(np.array(noiselabels), size=1)
                    # self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)


    def asymmetric_noise(self, noise_file):
        # self.train_labels_gt = self.train_labels.copy()
        if noise_file is not None:
            noise_label = json.load(open(noise_file, "r"))
            noisylabel = np.array(noise_label)[self.indexs]
            self.train_targets = np.array(noisylabel)
        else:
            for i in range(self.num_classes):
                indices = np.where(self.train_targets == i)[0]
                np.random.shuffle(indices)
                for j, idx in enumerate(indices):
                    if j < self.cfg_trainer['percent'] * len(indices):
                        self.noise_indx.append(idx)
                        # truck -> automobile
                        if i == 9:
                            self.train_targets[idx] = 1
                        # bird -> airplane
                        elif i == 2:
                            self.train_targets[idx] = 0
                        # cat -> dog
                        elif i == 3:
                            self.train_targets[idx] = 5
                        # dog -> cat
                        elif i == 5:
                            self.train_targets[idx] = 3
                        # deer -> horse
                        elif i == 4:
                            self.train_targets[idx] = 7


    def instance_noise(self, noise_file):

        noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))

        noisylabel = np.array(noise_label)[self.indexs]
        # truelabel = noise_label['clean_label_train'][self.indexs]

        self.train_targets = np.array(noisylabel)

    def contamination_noise(self, noise_file):

        noise_label = json.load(open(noise_file,"r"))
        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))

        noisylabel = np.array(noise_label)[self.indexs]
        # truelabel = noise_label['clean_label_train'][self.indexs]

        self.train_targets = np.array(noisylabel)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_imgs[index], int(self.train_targets[index]), int(self.train_targets_gt[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform_aug is not None:
            img2 = self.transform_aug(img)
        else:
            img2 = self.transform(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img2, target, index, target_gt

    def __len__(self):
        return len(self.train_imgs)

class mnist_val(MNIST):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(mnist_val, self).__init__(root, train=train,
                                          transform=transform, target_transform=target_transform,
                                          download=download)

        # self.train_data = self.data[indexs]
        # self.train_labels = np.array(self.targets)[indexs]
        self.num_classes = 10
        self.cfg_trainer = cfg_trainer
        if train:
            self.train_imgs = self.data[indexs]
            self.train_targets = np.array(self.targets)[indexs]
        else:
            self.train_imgs = self.data
            self.train_targets = np.array(self.targets)
        self.train_targets_gt = self.train_targets.copy()
        self.indexs = indexs

    def symmetric_noise(self, noise_file):
        if noise_file is not None:
            noise_label = json.load(open(noise_file, "r"))
            noisylabel = np.array(noise_label)[self.indexs]
            self.train_targets = np.array(noisylabel)
        else:
            indices = np.random.permutation(len(self.train_imgs))
            for i, idx in enumerate(indices):
                if i < self.cfg_trainer['percent'] * len(self.train_imgs):
                    noiselabels = list(range(self.num_classes))
                    noiselabels.remove(self.train_targets[idx])
                    self.train_targets[idx] = np.random.choice(np.array(noiselabels), size=1)
                    # self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)


    def instance_noise(self, noise_file):

        self.train_targets_gt = self.train_targets.copy()

        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))
        noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))

        noisylabel = np.array(noise_label)[self.indexs]

        self.train_targets = np.array(noisylabel)

    def contamination_noise(self, noise_file):
        self.train_targets_gt = self.train_targets.copy()

        # noise_label = torch.load(self.root + 'cifar-noisy/IDN_{:.1f}_C10.pt'.format(self.cfg_trainer['percent']))
        noise_label = json.load(open(noise_file,"r"))

        noisylabel = np.array(noise_label)[self.indexs]

        self.train_targets = np.array(noisylabel)

    def __len__(self):
        return len(self.train_imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, target_gt = self.train_imgs[index], int(self.train_targets[index]), int(self.train_targets_gt[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, target_gt


class MNISTDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True,
                 num_workers=4, pin_memory=True):
        config = ConfigParser.get_instance()
        cfg_trainer = config['trainer']

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        if config['train_loss']['args']['ratio_consistency'] > 0:
            transform_train_aug = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

            if config['data_augmentation']['type'] is not None:

                autoaug = transforms.Compose([])
                # if isinstance(cfg_trainer['aug'], list):
                #     autoaug.transforms.insert(0, Augmentation(C.get()['aug']))
                # else:
                if config['data_augmentation']['type'] == 'fa_reduced_cifar10':
                    autoaug.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
                elif config['data_augmentation']['type'] == 'autoaug_cifar10':
                    autoaug.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
                elif config['data_augmentation']['type'] == 'autoaug_extend':
                    autoaug.transforms.insert(0, Augmentation(autoaug_policy()))
                elif config['data_augmentation']['type'] == 'default':
                    pass
                else:
                    raise ValueError('not found augmentations. %s' % config['data_augmentation']['type'])
                transform_train_aug.transforms.insert(0, autoaug)

                if config['data_augmentation']['cutout'] > 0:
                    transform_train_aug.transforms.append(CutoutDefault(config['data_augmentation']['cutout']))
        else:
            transform_train_aug = None

        self.data_dir = data_dir

        noise_file = cfg_trainer['noise_file']

        self.train_dataset, self.val_dataset = get_mnist(config['data_loader']['args']['data_dir'], cfg_trainer,
                                                           train=training,
                                                           transform_train=transform_train,
                                                           transform_train_aug=transform_train_aug,
                                                           transform_val=transform_val, noise_file=noise_file, download=True)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset=self.val_dataset)

    # def run_loader(self, batch_size, shuffle, validation_split, num_workers, pin_memory):
    #     super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
    #                      val_dataset = self.val_dataset)

    def run_loader(self):
        pass