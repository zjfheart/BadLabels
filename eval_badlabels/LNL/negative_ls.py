"""
Original Code is from https://github.com/UCSC-REAL/negative-label-smoothing
"""

# -*- coding:utf-8 -*-
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from loader.dataloader import *
import argparse, sys
import numpy as np
import datetime
import shutil
from random import sample 
from generic.loss import loss_gls
from torch.utils.data import RandomSampler
sys.path.append("..")
from models.PreResNet import PreActResNet18
from models.densenet import *

parser = argparse.ArgumentParser(description="Pytorch Negative Label Smoothing")
parser.add_argument('--warmup-lr', type = float, default = 0.05)
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--loss', type = str, default = 'gls')
parser.add_argument('--result-dir', type = str, help = 'dir to save result txt files', default = '../export/nls')
parser.add_argument('--noise-rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--noise-type', type = str, help='[asym, sym]', default='sym')
parser.add_argument('--top-bn', action='store_true')
parser.add_argument('--ideal', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--net', type = str, help = 'cnn,resnet', default = 'preact_resnet18')
parser.add_argument('--warmup-epoch', type=int, default=120)
parser.add_argument('--n-epoch', type=int, default=100)
parser.add_argument('--wa', type=float, default=0)
parser.add_argument('--wb', type=float, default=1)
parser.add_argument('--smooth-rate', type=float, default=-1.0)
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--num-workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--data-path', default='../../data', type=str, help='path to dataset')
parser.add_argument('--noise-file', default=None)
parser.add_argument('--log', default='eval_results', type=str)



# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch, lr_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=lr_plan[epoch]
        

# Train the Model
def warmup(epoch, num_classes, train_loader, model, optimizer):
    train_total=0
    train_correct=0
    
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        logits = model(images)
            
        loss = loss_gls(epoch,logits, labels, 0.0, 1, 0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                  %(epoch+1, args.warmup_epoch, i+1, len(train_dataset)//batch_size, loss.data))


    train_acc=0.0
    return train_acc


def train(epoch, num_classes, train_loader, model, optimizer, smooth_rate, wa, wb):
    train_total = 0
    train_correct = 0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        batch_size = len(ind)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        logits = model(images)

        loss = loss_gls(epoch, logits, labels, smooth_rate, wa, wb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f'
                  % (epoch + 1, args.n_epoch, i + 1, len(train_dataset) // batch_size, loss.data))

    train_acc = 0.0
    return train_acc

# Evaluate the Model
def evaluate(test_loader,model,save=False,epoch=0,best_acc_=0,args=None):
    model.eval()    # Change model to 'eval' mode.
    print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total) 

    if save:
        if acc > best_acc_:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            torch.save(state,os.path.join(save_dir,args.log[4:]+'_best.pth.tar'))
            best_acc_ = acc
        if epoch == args.n_epoch -1:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            torch.save(state,os.path.join(save_dir,args.log[4:]+'_last.pth.tar'))

    return acc, best_acc_



#####################################main code ################################################
args = parser.parse_args()
# Seed
torch.cuda.set_device(args.gpu)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Hyper Parameters
batch_size = 128

wa_val = args.wa
wb_val = args.wb
smooth_rate_val = args.smooth_rate
n_type = args.noise_type

test_log = open('../eval_results/%s/%s/r%.1f/%s'%(args.dataset, args.net, args.noise_rate, args.log)+'.log', 'a')
test_log.write('===========================================\n')
test_log.write('Eval with Negative Label Smoothing ..\n')
test_log.flush()

# load dataset
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
    num_classes = 10
    num_training_samples = 50000
    grayscale = False

    train_dataset = cifar_dataset(dataset=args.dataset,
                                  noise_mode=args.noise_type,
                                  r=args.noise_rate,
                                  root_dir=args.data_path,
                                  transform=transform_train,
                                  mode="all",
                                  noise_file=args.noise_file)
    test_dataset = cifar_dataset(dataset=args.dataset,
                                 noise_mode=args.noise_type,
                                 r=args.noise_rate,
                                 root_dir=args.data_path,
                                 transform=transform_test,
                                 mode='test')
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
    num_classes = 100
    num_training_samples = 50000
    grayscale = False

    train_dataset = cifar_dataset(dataset=args.dataset,
                                  noise_mode=args.noise_type,
                                  r=args.noise_rate,
                                  root_dir=args.data_path,
                                  transform=transform_train,
                                  mode="all",
                                  noise_file=args.noise_file)
    test_dataset = cifar_dataset(dataset=args.dataset,
                                 noise_mode=args.noise_type,
                                 r=args.noise_rate,
                                 root_dir=args.data_path,
                                 transform=transform_test,
                                 mode='test')
elif args.dataset == 'mnist':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    num_classes = 10
    num_training_samples = 60000
    grayscale = True

    train_dataset = mnist_dataset(noise_mode=args.noise_type,
                                  r=args.noise_rate,
                                  root_dir=args.data_path,
                                  transform=transform_train,
                                  mode="all",
                                  noise_file=args.noise_file)
    test_dataset = mnist_dataset(noise_mode=args.noise_type,
                                 r=args.noise_rate,
                                 root_dir=args.data_path,
                                 transform=transform_test,
                                 mode='test')

# load model
print('building model...')
# if args.model == 'resnet34':
#     model = ResNet34(num_classes)
if args.net == 'preact_resnet18':
    model = PreActResNet18(num_classes, grayscale)
elif args.net == 'densenet':
    model = DenseNet3(depth=40, num_classes=num_classes)
else:
    raise NotImplementedError
print('building model done')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


### save result and model checkpoint #######   
save_dir = args.result_dir + '/' + args.dataset + '/' + args.net + '/r' + str(args.noise_rate)
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = 128, 
                                   num_workers=args.num_workers,
                                   shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64, 
                                  num_workers=args.num_workers,
                                  shuffle=False)
warmup_lr_plan = [0.1] * 40 + [0.01] * 40 + [0.001] * 40
lr_plan = [1e-6] * 100

model.cuda()
# txtfile=save_dir + '/' +  args.loss + args.noise_type + str(args.noise_rate) + '.txt'
# if os.path.exists(txtfile):
#     os.system('rm %s' % txtfile)
# with open(txtfile, "a") as myfile:
#     myfile.write('epoch: test_acc \n')

epoch=0
train_acc = 0
best_acc_ = 0.0


for epoch in range(args.warmup_epoch):
# train models
    adjust_learning_rate(optimizer, epoch, warmup_lr_plan)
    model.train()
    train_acc = warmup(epoch,num_classes,train_loader, model, optimizer)

# evaluate models
    test_acc, best_acc_ = evaluate(test_loader=test_loader, save=True, model=model,epoch=epoch,best_acc_=best_acc_,args=args)

    print('test acc on test images is ', test_acc)
    # with open(txtfile, "a") as myfile:
    #     myfile.write(str(int(epoch)) + ': ' + str(test_acc) + "\n")
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, test_acc))
    test_log.flush()


# main training
if args.net == 'preact_resnet18':
    model = PreActResNet18(num_classes, grayscale)
elif args.net == 'densenet':
    model = DenseNet3(depth=40, num_classes=num_classes)
else:
    raise NotImplementedError
state_dict = torch.load(os.path.join(save_dir,args.log[4:]+'_best.pth.tar'), map_location = "cpu")
model.load_state_dict(state_dict['state_dict'])
print('re-load model done')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.0001, nesterov= True)
model.cuda()

epoch = 0
train_acc = 0
best_acc_ = 0.0

for epoch in range(args.n_epoch):
    # train models
    adjust_learning_rate(optimizer, epoch, lr_plan)
    model.train()
    train_acc = train(epoch, num_classes, train_loader, model, optimizer, smooth_rate=smooth_rate_val, wa=wa_val,
                      wb=wb_val)

    # evaluate models
    test_acc, best_acc_ = evaluate(test_loader=test_loader, save=False, model=model, epoch=epoch, best_acc_=best_acc_,
                                   args=args)

    overall_epoch = args.warmup_epoch + epoch
    print(f'epoch{overall_epoch}, test acc on test images is {test_acc}')
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (overall_epoch, test_acc))
    test_log.flush()

test_log.close()