"""
Original Code is from https://github.com/LiJunnan1992/DivideMix
"""
import gzip
import os
import random

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import pandas as pd
from torchnet.meter import AUCMeter
import sys
import numpy as np

sys.path.append("../..")
from augment.randaug import *


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class mnist_dataset(Dataset):
    def __init__(self, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], p_pred=None):

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode

        if self.mode == 'test':
            with gzip.open('%s/MNIST/raw/t10k-labels-idx1-ubyte.gz' % root_dir, 'rb') as lbpath:
                self.test_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            with gzip.open('%s/MNIST/raw/t10k-images-idx3-ubyte.gz' % root_dir, 'rb') as imgpath:
                self.test_data = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(self.test_label), 28, 28)
        else:
            with gzip.open('%s/MNIST/raw/train-labels-idx1-ubyte.gz' % root_dir, 'rb') as lbpath:
                train_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            with gzip.open('%s/MNIST/raw/train-images-idx3-ubyte.gz' % root_dir, 'rb') as imgpath:
                train_data = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_label), 28, 28)
            self.train_label = train_label

            if noise_file is not None:
                if noise_file[-5:] == '.json':
                    noise_label = json.load(open(noise_file, "r"))
                elif noise_file[-4:] == '.csv':
                    noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
                else:
                    raise NotImplementedError
                print(f'load noise labels from {noise_file} ..')
            else:  # inject noise
                noise_label = []
                num_labels = len(train_label)
                idx = list(range(num_labels))
                random.shuffle(idx)
                num_noise = int(self.r * num_labels)
                noise_idx = idx[:num_noise]
                for i in range(num_labels):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            noiselabels = list(range(10))
                            noiselabels.remove(train_label[i])
                            noiselabel = random.choice(noiselabels)
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(int(train_label[i]))

            clean_or_not = (np.array(noise_label) == np.array(self.train_label))

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if p_pred is not None:
                    peturbed_idx = p_pred.nonzero()[0]
                else:
                    peturbed_idx = np.array([])
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    for i in peturbed_idx:
                        if i in pred_idx:
                            pred_idx = np.delete(pred_idx, np.where(pred_idx == i)[0], axis=0)
                    self.probability = [probability[i] for i in pred_idx]

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]
                    for i in peturbed_idx:
                        if i not in pred_idx:
                            pred_idx = np.append(pred_idx, [i], axis=0)

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                self.train_label = [self.train_label[i] for i in pred_idx]
                num_clean = 0
                for i, c in enumerate(clean_or_not):
                    if i in pred_idx:
                        if c: num_clean += 1
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))
                if len(self.noise_label) != 0:
                    print(f"num of clean data is {num_clean} ({num_clean / len(self.noise_label) * 100.:.2f}%)")

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], int(self.noise_label[index]), self.probability[index]
            img = Image.fromarray(img, mode='L')
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img, mode='L')
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode == 'all':
            img, target = self.train_data[index], int(self.noise_label[index])
            img = Image.fromarray(img, mode='L')
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], int(self.test_label[index])
            img = Image.fromarray(img, mode='L')
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class mnist_dataloader():
    def __init__(self, r, noise_mode, batch_size, num_workers, root_dir, noise_file=''):
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def run(self, mode, pred=[], prob=[], p_pred=None):
        if mode == 'warmup':
            all_dataset = mnist_dataset(noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                        noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = mnist_dataset(noise_mode=self.noise_mode, r=self.r,
                                            root_dir=self.root_dir, transform=self.transform_train, mode="labeled",
                                            noise_file=self.noise_file, pred=pred, probability=prob, p_pred=p_pred)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = mnist_dataset(noise_mode=self.noise_mode, r=self.r,
                                              root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",
                                              noise_file=self.noise_file, pred=pred, p_pred=p_pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = mnist_dataset(noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = mnist_dataset(noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                         noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

        elif mode == 'peer':
            all_dataset = mnist_dataset(noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                        noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            )
            return trainloader


class mnist_dataset_split(Dataset):
    def __init__(self, r, noise_mode, root_dir, transform, mode, noise_file='', seed=17, split_percent=0.9):

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise

        np.random.seed(seed)

        if self.mode == 'test':
            with gzip.open('%s/MNIST/raw/t10k-labels-idx1-ubyte.gz' % root_dir, 'rb') as lbpath:
                self.test_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            with gzip.open('%s/MNIST/raw/t10k-images-idx3-ubyte.gz' % root_dir, 'rb') as imgpath:
                self.test_data = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(self.test_label), 28, 28)
        else:
            with gzip.open('%s/MNIST/raw/train-labels-idx1-ubyte.gz' % root_dir, 'rb') as lbpath:
                train_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
            with gzip.open('%s/MNIST/raw/train-images-idx3-ubyte.gz' % root_dir, 'rb') as imgpath:
                train_data = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_label), 28, 28)
            self.train_label = train_label

            if noise_file is not None:
                if noise_file[-5:] == '.json':
                    noise_label = json.load(open(noise_file, "r"))
                elif noise_file[-4:] == '.csv':
                    noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
                else:
                    raise NotImplementedError
                print(f'load noise labels from {noise_file} ..')
            else:  # inject noise
                noise_label = []
                num_labels = len(train_label)
                idx = list(range(num_labels))
                random.shuffle(idx)
                num_noise = int(self.r * num_labels)
                noise_idx = idx[:num_noise]
                for i in range(num_labels):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            noiselabels = list(range(10))
                            noiselabels.remove(train_label[i])
                            noiselabel = random.choice(noiselabels)
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(int(train_label[i]))

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                indice = np.random.choice(60000, int(60000 * split_percent), replace=False)
                indice_val = np.delete(np.arange(60000), indice)
                if self.mode == 'train':
                    self.train_data = train_data[indice]
                    self.noise_label = [noise_label[i] for i in indice]
                elif self.mode == 'val':
                    self.train_data = train_data[indice_val]
                    self.noise_label = [noise_label[i] for i in indice_val]

    def __getitem__(self, index):
        if self.mode == 'all' or self.mode == 'train' or self.mode == 'val':
            img, target = self.train_data[index], int(self.noise_label[index])
            img = Image.fromarray(img, mode='L')
            img = self.transform(img)
            return img, target

        elif self.mode == 'test':
            img, target = self.test_data[index], int(self.test_label[index])
            img = Image.fromarray(img, mode='L')
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[],
                 p_pred=None):

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/cifar-10-batches-py/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/cifar-100-python/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/cifar-10-batches-py/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/cifar-100-python/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            self.train_label = train_label

            if noise_file is not None:
                if noise_file[-5:] == '.json':
                    noise_label = json.load(open(noise_file, "r"))
                elif noise_file[-4:] == '.csv':
                    noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
                else:
                    raise NotImplementedError
                print(f'load noise labels from {noise_file} ..')
            else:  # inject noise
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r * 50000)
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            if dataset == 'cifar10':
                                noiselabels = list(range(10))
                                noiselabels.remove(train_label[i])
                                noiselabel = random.choice(noiselabels)
                                # noiselabel = random.randint(0, 9)
                            elif dataset == 'cifar100':
                                noiselabels = list(range(100))
                                noiselabels.remove(train_label[i])
                                noiselabel = random.choice(noiselabels)
                                # noiselabel = random.randint(0, 99)
                            noise_label.append(noiselabel)
                        elif noise_mode == 'asym':
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])

            clean_or_not = (np.array(noise_label) == np.array(self.train_label))

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if p_pred is not None:
                    peturbed_idx = p_pred.nonzero()[0]
                else:
                    peturbed_idx = np.array([])
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    for i in peturbed_idx:
                        if i in pred_idx:
                            pred_idx = np.delete(pred_idx, np.where(pred_idx == i)[0], axis=0)
                    self.probability = [probability[i] for i in pred_idx]

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]
                    for i in peturbed_idx:
                        if i not in pred_idx:
                            pred_idx = np.append(pred_idx, [i], axis=0)

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]
                self.train_label = [self.train_label[i] for i in pred_idx]
                num_clean = 0
                for i, c in enumerate(clean_or_not):
                    if i in pred_idx:
                        if c: num_clean += 1
                print("%s data has a size of %d" % (self.mode, len(self.noise_label)))
                if len(self.noise_label) != 0:
                    print(f"num of clean data is {num_clean} ({num_clean / len(self.noise_label) * 100.:.2f}%)")

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.num_train_labels = 50000
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode, pred=[], prob=[], p_pred=None):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                        noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                            root_dir=self.root_dir, transform=self.transform_train, mode="labeled",
                                            noise_file=self.noise_file, pred=pred, probability=prob, p_pred=p_pred)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                              root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled",
                                              noise_file=self.noise_file, pred=pred, p_pred=p_pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                         noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader

        elif mode == 'peer':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                        noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            )
            return trainloader


class cifar_dataset_split(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', seed=17, split_percent=0.9):

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise

        np.random.seed(seed)

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/cifar-10-batches-py/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/cifar-100-python/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/cifar-10-batches-py/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/cifar-100-python/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if noise_file is not None:
                if noise_file[-5:] == '.json':
                    noise_label = json.load(open(noise_file, "r"))
                elif noise_file[-4:] == '.csv':
                    noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
                else:
                    raise NotImplementedError
                print('load noise labels ..')
            else:  # inject noise
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r * 50000)
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            if dataset == 'cifar10':
                                noiselabels = list(range(10))
                                noiselabels.remove(train_label[i])
                                noiselabel = random.choice(noiselabels)
                                # noiselabel = random.randint(0, 9)
                            elif dataset == 'cifar100':
                                noiselabels = list(range(100))
                                noiselabels.remove(train_label[i])
                                noiselabel = random.choice(noiselabels)
                                # noiselabel = random.randint(0, 99)
                            noise_label.append(noiselabel)
                        elif noise_mode == 'asym':
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                indice = np.random.choice(50000, int(50000 * split_percent), replace=False)
                indice_val = np.delete(np.arange(50000), indice)
                if self.mode == 'train':
                    self.train_data = train_data[indice]
                    self.noise_label = [noise_label[i] for i in indice]
                elif self.mode == 'val':
                    self.train_data = train_data[indice_val]
                    self.noise_label = [noise_label[i] for i in indice_val]

    def __getitem__(self, index):
        if self.mode == 'all' or self.mode == 'train' or self.mode == 'val':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class clothing_dataset(Dataset):
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14):

        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}

        with open('%s/noisy_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/images/' % self.root + entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        with open('%s/clean_label_kv.txt' % self.root, 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/images/' % self.root + entry[0][7:]
                self.test_labels[img_path] = int(entry[1])

        if mode == 'all':
            train_imgs = []
            with open('%s/noisy_train_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/images/' % self.root + l[7:]
                    train_imgs.append(img_path)
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]
                if class_num[label] < (num_samples / 14) and len(self.train_imgs) < num_samples:
                    self.train_imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.train_imgs)
        elif self.mode == "labeled":
            train_imgs = paths
            pred_idx = pred.nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))
        elif self.mode == "unlabeled":
            train_imgs = paths
            pred_idx = (1 - pred).nonzero()[0]
            self.train_imgs = [train_imgs[i] for i in pred_idx]
            self.probability = [probability[i] for i in pred_idx]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

        elif mode == 'test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/images/' % self.root + l[7:]
                    self.test_imgs.append(img_path)
        elif mode == 'val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/images/' % self.root + l[7:]
                    self.val_imgs.append(img_path)

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, img_path
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_imgs)
        if self.mode == 'val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)


class clothing_dataloader():
    def __init__(self, root, batch_size, num_batches, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.root = root

        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])

    def run(self, mode, pred=[], prob=[], paths=[]):
        if mode == 'warmup':
            warmup_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='all',
                                              num_samples=self.num_batches * self.batch_size * 2)
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers)
            return warmup_loader
        elif mode == 'train':
            labeled_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='labeled', pred=pred,
                                               probability=prob, paths=paths)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            unlabeled_dataset = clothing_dataset(self.root, transform=self.transform_train, mode='unlabeled', pred=pred,
                                                 probability=prob, paths=paths)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)
            return labeled_loader, unlabeled_loader
        elif mode == 'eval_train':
            eval_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='all',
                                            num_samples=self.num_batches * self.batch_size)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
        elif mode == 'test':
            test_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader
        elif mode == 'val':
            val_dataset = clothing_dataset(self.root, transform=self.transform_test, mode='val')
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=self.num_workers)
            return val_loader


class cifarn_dataset(Dataset):
    def __init__(self, dataset, noise_type, noise_path, root_dir, transform, mode, pred=[], probability=[]):
        self.dataset = dataset
        self.transform = transform
        self.mode = mode
        self.noise_type = noise_type
        self.noise_path = noise_path

        if dataset == 'cifar10':
            self.nb_classes = 10
        elif dataset == 'cifar100':
            self.nb_classes = 100
        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/cifar-10-batches-py/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/cifar-100-python/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/cifar-10-batches-py/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/cifar-100-python/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            self.train_labels = train_label

            # if noise_type is not None:
            if noise_type != 'clean':
                # Load human noisy labels
                train_noisy_labels = self.load_label()
                self.train_noisy_labels = train_noisy_labels.tolist()
            noise_label = train_noisy_labels

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]

    def load_label(self):
        # NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        print(f'load cifar-n-{self.noise_type} labels from {self.noise_path} ..')
        if isinstance(noise_label, dict):
            return noise_label[self.noise_type].reshape(-1)
        else:
            raise Exception('Input Error')

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifarn_dataloader():
    def __init__(self, dataset, noise_type, noise_path, batch_size, num_workers, root_dir):
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            all_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                         root_dir=self.root_dir, transform=self.transform_train, mode="all")
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)
            # never show noisy rate again
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type,
                                             noise_path=self.noise_path,
                                             root_dir=self.root_dir, transform=self.transform_train, mode="labeled",
                                             pred=pred, probability=prob)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            unlabeled_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                              root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)

            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifarn_dataset(dataset=self.dataset, noise_type=self.noise_type, noise_path=self.noise_path,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
        # never print again

