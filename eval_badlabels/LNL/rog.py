"""
Original Code is from https://github.com/pokaxpoka/RoGNoisyLabel
"""

from __future__ import print_function
import argparse
import json
import random

import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import loader.rog_loader as data_loader
import generic.rog_utils as utils
import numpy as np
import os, sys

from torchvision import datasets, transforms
from torch.autograd import Variable

sys.path.append("..")
from models.PreResNet_rog import *
from models.densenet_rog import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch RoG')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100')
parser.add_argument('--data-path', default='../../data', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing-lr', default='40,80', help='decreasing strategy')
parser.add_argument('--net', default='preact_resnet18', help="Type of Classification Nets")
parser.add_argument('--optimizer-flag', default='sgd', help="Type of optimizer")
parser.add_argument('--numclass', type=int, default=10, help='the # of classes')
parser.add_argument('--noise-fraction', type=float, default=0.2, help='noisy fraction')
parser.add_argument('--noise-type', default='sym', help='type_of_noise')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')

parser.add_argument('--noise-file', default=None)
parser.add_argument('--log', type=str, default='eval_results')
parser.add_argument('--export', default='../export/rog', type=str)
args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

test_log = open('../eval_results/%s/%s/r%.1f/%s'%(args.dataset, args.net, args.noise_fraction, args.log)+'.log', 'a')
test_log.write('===========================================\n')
test_log.write('Eval with RoG ..\n')
test_log.flush()

""" Pre-Train """

print('load data: ', args.dataset)
if args.dataset == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
elif args.dataset == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    args.numclass = 100
    args.decreasing_lr = '80,120,160'
    args.epochs = 200
elif args.dataset == 'mnist':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

train_loader, _ = data_loader.getTargetDataSet(args.dataset, args.batch_size, transform_train, args.data_path)
_, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, transform_test, args.data_path)

if args.noise_fraction > 0:
    print('load noisy labels')

    noise_file = args.noise_file
    if noise_file[-5:] == '.json':
        noise_label = json.load(open(noise_file, "r"))
    elif noise_file[-4:] == '.csv':
        noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
    else:
        raise NotImplementedError
        
    # new_label = torch.load(args.label_root)
    train_loader.dataset.targets = noise_label
    
print('Model: ', args.net)
grayscale = True if args.dataset == 'mnist' else False
if args.net == 'preact_resnet18':
    model = PreActResNet18(num_classes=args.numclass, grayscale=grayscale)
elif args.net == 'densenet':
    model = DenseNet3(depth=40, num_classes=args.numclass)
else:
    raise NotImplementedError

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

def train(epoch):
    model.train()
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)                
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), total,
                100. * batch_idx / total, loss.item()))

def test(epoch):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    for data, target in test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data), dim=1)
        test_loss += F.nll_loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,100. * correct / total))
    test_log.flush()

test_log.write('Pre-Training ..\n')
test_log.flush()
for epoch in range(1, args.epochs + 1):
    train(epoch)
    if epoch in decreasing_lr:
        optimizer.param_groups[0]['lr'] *= args.droprate
    test(epoch)
    
# args.outf = args.outf + '/' + args.net_type + '/' + args.dataset + '/' + args.noise_type + '/' + str(args.noise_fraction) + '/'
args.outf = f"{args.export}/{args.dataset}/{args.net}/r{args.noise_fraction}/{args.log[4:]}"

if not os.path.isdir(args.outf):
    os.makedirs(args.outf)
torch.save(model.state_dict(), '%s/model.pth' % (args.outf))


""" Inference """
batch_size = 200

num_output = 2

layer_list = list(range(num_output))

print('load dataset: ' + args.dataset)
num_classes = 10
if args.dataset == 'cifar10':
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
elif args.dataset == 'cifar100':
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    num_classes = 100
elif args.dataset == 'mnist':
    in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, batch_size, in_transform, args.data_path, False)

if args.noise_fraction > 0:
    print('load noisy labels')
    noise_file = args.noise_file
    if noise_file[-5:] == '.json':
        noise_label = json.load(open(noise_file, "r"))
    elif noise_file[-4:] == '.csv':
        noise_label = list(pd.read_csv(noise_file)['label_noisy'].values.astype(int))
    else:
        raise NotImplementedError

    # new_label = torch.load(args.label_root)
    train_loader.dataset.targets = noise_label

num_val = 50
if args.dataset == 'cifar100':
    num_val = 5
total_train_data, total_train_label, _, _ = data_loader.get_raw_data(train_loader, num_classes, 0)

total_test_data, total_test_label, val_index, test_index \
    = data_loader.get_raw_data(test_loader, num_classes, num_val)

total_val_data, total_val_label, total_test_data, total_test_label \
    = data_loader.get_validation(total_test_data, total_test_label, val_index, test_index)

print('extact features')
utils.extract_features(model, total_train_data, total_train_label, args.outf, "train_val", grayscale)
utils.extract_features(model, total_val_data, total_val_label, args.outf, "test_val", grayscale)
utils.extract_features(model, total_test_data, total_test_label, args.outf, "test_test", grayscale)

test_data_list, test_label_list = [], []
test_val_data_list, test_val_label_list = [], []
train_data_list, train_label_list = [], []

for layer in layer_list:
    file_name_data = '%s/test_test_feature_%s.npy' % (args.outf, str(layer))
    file_name_label = '%s/test_test_label.npy' % (args.outf)

    test_data = torch.from_numpy(np.load(file_name_data)).float()
    test_label = torch.from_numpy(np.load(file_name_label)).long()
    test_data_list.append(test_data)

    file_name_data = '%s/test_val_feature_%s.npy' % (args.outf, str(layer))
    file_name_label = '%s/test_val_label.npy' % (args.outf)

    test_data_val = torch.from_numpy(np.load(file_name_data)).float()
    test_label_val = torch.from_numpy(np.load(file_name_label)).long()
    test_val_data_list.append(test_data_val)
    file_name_data = '%s/train_val_feature_%s.npy' % (args.outf, str(layer))
    file_name_label = '%s/train_val_label.npy' % (args.outf)

    if args.noise_type == 'sym':
        for index in range(int(test_label_val.size(0) * args.noise_fraction)):
            prev_label = test_label_val[index]
            while (True):
                new_label = np.random.randint(0, num_classes)
                if prev_label != new_label:
                    test_label_val[index] = new_label
                    break;

    elif args.noise_type == 'asym':
        for index in range(int(test_label_val.size(0) * args.noise_fraction)):
            prev_label = test_label_val[index]
            new_label = (prev_label + 1) % num_classes
            test_label_val[index] = new_label

    train_data = torch.from_numpy(np.load(file_name_data)).float()
    train_label = torch.from_numpy(np.load(file_name_label)).long()

    train_data_list.append(train_data)
    train_label_list.append(train_label)

test_label_list.append(test_label)
test_val_label_list.append(test_label_val)

print('Random Sample Mean')
sample_mean_list, sample_precision_list = [], []
for index in range(len(layer_list)):
    sample_mean, sample_precision, _ = \
        utils.random_sample_mean(train_data_list[index].cuda(), train_label_list[index].cuda(), num_classes)
    sample_mean_list.append(sample_mean)
    sample_precision_list.append(sample_precision)

print('Single MCD and merge the parameters')
new_sample_mean_list = []
new_sample_precision_list = []
for index in range(len(layer_list)):
    new_sample_mean = torch.Tensor(num_classes, train_data_list[index].size(1)).fill_(0).cuda()
    new_covariance = 0
    for i in range(num_classes):
        index_list = train_label_list[index].eq(i)
        temp_feature = train_data_list[index][index_list.nonzero(), :]
        temp_feature = temp_feature.view(temp_feature.size(0), -1)
        temp_mean, temp_cov, _ \
            = utils.MCD_single(temp_feature.cuda(), sample_mean_list[index][i], sample_precision_list[index])
        new_sample_mean[i].copy_(temp_mean)
        if i == 0:
            new_covariance = temp_feature.size(0) * temp_cov
        else:
            new_covariance += temp_feature.size(0) * temp_cov

    new_covariance = new_covariance / train_data_list[index].size(0)
    new_precision = scipy.linalg.pinvh(new_covariance)
    new_precision = torch.from_numpy(new_precision).float().cuda()
    new_sample_mean_list.append(new_sample_mean)
    new_sample_precision_list.append(new_precision)

G_soft_list = []
target_mean = new_sample_mean_list
target_precision = new_sample_precision_list
for i in range(len(new_sample_mean_list)):
    dim_feature = new_sample_mean_list[i].size(1)
    sample_w = torch.mm(target_mean[i], target_precision[i])
    sample_b = -0.5 * torch.mm(torch.mm(target_mean[i], target_precision[i]), \
                               target_mean[i].t()).diag() + torch.Tensor(num_classes).fill_(
        np.log(1. / num_classes)).cuda()
    G_soft_layer = nn.Linear(int(dim_feature), num_classes).cuda()
    G_soft_layer.weight.data.copy_(sample_w)
    G_soft_layer.bias.data.copy_(sample_b)
    G_soft_list.append(G_soft_layer)

print('Construct validation set')
sel_index = -1
selected_list = utils.make_validation(test_val_data_list[sel_index], test_val_label_list[-1], \
                                      target_mean[sel_index], target_precision[sel_index], num_classes)
new_val_data_list = []
for i in range(len(new_sample_mean_list)):
    new_val_data = torch.index_select(test_val_data_list[i], 0, selected_list.cpu())
    new_val_label = torch.index_select(test_val_label_list[-1], 0, selected_list.cpu())
    new_val_data_list.append(new_val_data)

soft_weight = utils.train_weights(G_soft_list, new_val_data_list, new_val_label)
soft_acc = utils.test_softmax(model, total_test_data, total_test_label)

RoG_acc = utils.test_ensemble(G_soft_list, soft_weight, test_data_list, test_label_list[-1])

print('softmax accuracy: ' + str(soft_acc.item()))
print('RoG accuracy: ' + str(RoG_acc.item()))
test_log.write('Inference ..\n')
test_log.write('softmax accuracy: %s\n'%(str(soft_acc.item())))
test_log.write('RoG accuracy: %s\n'%(str(RoG_acc.item())))
test_log.flush()
test_log.close()
