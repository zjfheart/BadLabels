"""
Original Code is from https://github.com/Justherozen/ProMix
"""
import gzip
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import pandas as pd
import copy
import sys
sys.path.append("../..")
from augment.randaug import *


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class promix_cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_type, root_dir, transform, mode, transform_s=None,
                 noise_file='',
                 pred=[], probability=[], probability2=[]):
        self.r = r
        self.dataset = dataset
        self.transform = transform
        self.transform_s = transform_s
        self.mode = mode
        self.noise_type = noise_type
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
        idx_each_class_noisy = [[] for i in range(10)]
        # self.print_show = print_show

        if dataset == 'cifar10':
            self.nb_classes = 10
            idx_each_class_noisy = [[] for i in range(10)]
        elif dataset == 'cifar100':
            self.nb_classes = 100
            idx_each_class_noisy = [[] for i in range(100)]
        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/cifar-10-batches-py/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/cifar-100-python/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/cifar-10-batches-py/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/cifar-100-python/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            self.train_labels = train_label

            if noise_file is not None:
                if noise_file[-5:] == '.json':
                    noise_label = json.load(open(noise_file,"r"))
                elif noise_file[-4:] == '.csv':
                    noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
                else:
                    raise NotImplementedError
                print('load noise labels ..')
            else:    #inject noise
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_type == 'sym':
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
                        elif noise_type=='asym':
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])

            if self.mode == 'all_lab':
                self.probability = probability
                self.probability2 = probability2
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(noise_label) == np.array(train_label))

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2
        elif self.mode == 'all_lab':
            img, target, prob, prob2 = self.train_data[index], self.noise_label[index], self.probability[index], \
                                       self.probability2[index]
            true_labels = self.train_labels[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob, prob2, true_labels, index
        elif self.mode == 'all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            if self.transform_s is not None:
                img1 = self.transform(img)
                img2 = self.transform_s(img)
                return img1, img2, target, index
            else:
                img = self.transform(img)
                return img, target, index
        elif self.mode == 'all2':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, index
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


class promix_cifar_dataloader():
    def __init__(self, dataset, r, noise_type, batch_size, num_workers, root_dir,
                 noise_file=''):
        self.dataset = dataset
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.r = r
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.transform_train_s = copy.deepcopy(self.transform_train)
            self.transform_train_s.transforms.insert(0, RandomAugment(3,5))
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
            self.transform_train_s = copy.deepcopy(self.transform_train)
            self.transform_train_s.transforms.insert(0, RandomAugment(3,5))
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
        self.print_show = True

    def run(self, mode, pred=[], prob=[],prob2=[]):
        if mode == 'warmup':
            all_dataset = promix_cifar_dataset(dataset=self.dataset, noise_type=self.noise_type, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_train,
                                         transform_s=self.transform_train_s, mode="all",
                                         noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            self.print_show = False
            # never show noisy rate again
            return trainloader

        elif mode == 'train':
            labeled_dataset = promix_cifar_dataset(dataset=self.dataset, noise_type=self.noise_type, r=self.r,
                                             root_dir=self.root_dir, transform=self.transform_train, mode="all_lab",
                                             noise_file=self.noise_file, pred=pred, probability=prob,probability2=prob2,
                                             transform_s=self.transform_train_s)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True)

            return labeled_trainloader

        elif mode == 'test':
            test_dataset = promix_cifar_dataset(dataset=self.dataset, noise_type=self.noise_type, r=self.r,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = promix_cifar_dataset(dataset=self.dataset, noise_type=self.noise_type, r=self.r,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                          noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
        # never print again


class promix_mnist_dataset(Dataset):
    def __init__(self, r, noise_type, root_dir, transform, mode, transform_s=None,
                 noise_file='',
                 pred=[], probability=[], probability2=[]):
        self.r = r
        self.transform = transform
        self.transform_s = transform_s
        self.mode = mode
        self.noise_type = noise_type
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
        # self.print_show = print_show

        self.nb_classes = 10
        idx_each_class_noisy = [[] for i in range(10)]

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
            self.train_labels = train_label

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
                        if noise_type == 'sym':
                            noiselabels = list(range(10))
                            noiselabels.remove(train_label[i])
                            noiselabel = random.choice(noiselabels)
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(int(train_label[i]))

            if self.mode == 'all_lab':
                self.probability = probability
                self.probability2 = probability2
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]

                    clean = (np.array(noise_label) == np.array(train_label))

                elif self.mode == "unlabeled":
                    pred_idx = (1 - pred).nonzero()[0]

                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, prob = self.train_data[index], int(self.noise_label[index]), self.probability[index]
            img = Image.fromarray(img, mode='L')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img, mode='L')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2
        elif self.mode == 'all_lab':
            img, target, prob, prob2 = self.train_data[index], int(self.noise_label[index]), self.probability[index], \
                                       self.probability2[index]
            true_labels = self.train_labels[index]
            img = Image.fromarray(img, mode='L')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, prob, prob2, true_labels, index
        elif self.mode == 'all':
            img, target = self.train_data[index], int(self.noise_label[index])
            img = Image.fromarray(img, mode='L')
            if self.transform_s is not None:
                img1 = self.transform(img)
                img2 = self.transform_s(img)
                return img1, img2, target, index
            else:
                img = self.transform(img)
                return img, target, index
        elif self.mode == 'all2':
            img, target = self.train_data[index], int(self.noise_label[index])
            img = Image.fromarray(img, mode='L')
            img1 = self.transform(img)
            img2 = self.transform_s(img)
            return img1, img2, target, index
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


class promix_mnist_dataloader():
    def __init__(self, r, noise_type, batch_size, num_workers, root_dir,
                 noise_file=''):
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.r = r
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.transform_train_s = copy.deepcopy(self.transform_train)
        self.transform_train_s.transforms.insert(0, RandomAugment_MNIST(3, 5))
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def run(self, mode, pred=[], prob=[],prob2=[]):
        if mode == 'warmup':
            all_dataset = promix_mnist_dataset(noise_type=self.noise_type, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_train,
                                         transform_s=self.transform_train_s, mode="all",
                                         noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            self.print_show = False
            # never show noisy rate again
            return trainloader

        elif mode == 'train':
            labeled_dataset = promix_mnist_dataset(noise_type=self.noise_type, r=self.r,
                                             root_dir=self.root_dir, transform=self.transform_train, mode="all_lab",
                                             noise_file=self.noise_file, pred=pred, probability=prob,probability2=prob2,
                                             transform_s=self.transform_train_s)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True)

            return labeled_trainloader

        elif mode == 'test':
            test_dataset = promix_mnist_dataset(noise_type=self.noise_type, r=self.r,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = promix_mnist_dataset(noise_type=self.noise_type, r=self.r,
                                          root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                          noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
        # never print again