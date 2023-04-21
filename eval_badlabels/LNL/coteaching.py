"""
Original Code is from https://github.com/bhanML/Co-teaching
"""

# -*- coding:utf-8 -*-
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable
import loader.dataloader as dataloader
from generic.loss import loss_coteaching
import argparse, sys
sys.path.append("..")
from models.PreResNet import PreActResNet18
from models.densenet import *
import numpy as np

parser = argparse.ArgumentParser(description="Pytorch Co-teaching")
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result-dir', type = str, help = 'dir to save result txt files', default = 'ctresults/')
parser.add_argument('--noise-rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget-rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise-type', type = str, help='[asym, sym]', default='sym')
parser.add_argument('--num-gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top-bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default = 'cifar10')
parser.add_argument('--n-epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--num-workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num-iter-per-epoch', type=int, default=400)
parser.add_argument('--epoch-decay-start', type=int, default=80)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--noise-file', default=None)
parser.add_argument('--log', type=str, default='eval_results')
parser.add_argument('--data-path', type=str, default='../../data')
parser.add_argument('--net', type=str, default='preact_resnet18')

args = parser.parse_args()

torch.cuda.set_device(args.gpu)

# Seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr

test_log = open('../eval_results/%s/%s/r%.1f/%s'%(args.dataset, args.net, args.noise_rate, args.log)+'.log', 'a')
test_log.write('===========================================\n')
test_log.write('Eval with Co-Teaching ..\n')
test_log.flush()

# load dataset
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    loader = dataloader.mnist_dataloader(r=args.noise_rate, noise_mode=args.noise_type, batch_size=batch_size,
                                         num_workers=5, root_dir=args.data_path, noise_file=args.noise_file)
    
if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    loader = dataloader.cifar_dataloader(args.dataset, r=args.noise_rate, noise_mode=args.noise_type, batch_size=batch_size,
                                         num_workers=5,
                                         root_dir=args.data_path, noise_file=args.noise_file)

if args.dataset=='cifar100':
    input_channel=3
    num_classes=100
    args.top_bn = False
    args.epoch_decay_start = 100
    args.n_epoch = 200
    loader = dataloader.cifar_dataloader(args.dataset, r=args.noise_rate, noise_mode=args.noise_type, batch_size=batch_size,
                                         num_workers=5,
                                         root_dir=args.data_path, noise_file=args.noise_file)

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

# noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2):
    print('Training ...')
    # pure_ratio_list=[]
    # pure_ratio_1_list=[]
    # pure_ratio_2_list=[]
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0 

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits1=model1(images)
        prec1, = accuracy(logits1, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(images)
        prec2, = accuracy(logits2, labels, topk=(1, ))
        train_total2+=1
        train_correct2+=prec2
        loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind)
        # pure_ratio_1_list.append(100*pure_ratio_1)
        # pure_ratio_2_list.append(100*pure_ratio_2)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_loader.dataset)//batch_size, prec1, prec2, loss_1.item(), loss_2.item()))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating ...')
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode 
    correct2 = 0
    total2 = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
 
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = loader.run('warmup')
    test_loader = loader.run('test')

    # Define models
    print('building model...')
    if args.net == 'preact_resnet18':
        if input_channel == 1:
            grayscale = True
        else:
            grayscale = False
        cnn1 = PreActResNet18(num_classes=num_classes, grayscale=grayscale)
        cnn1.cuda()
        print(cnn1.parameters)
        optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)

        cnn2 = PreActResNet18(num_classes=num_classes, grayscale=grayscale)
        cnn2.cuda()
        print(cnn2.parameters)
        optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)
    elif args.net == 'densenet':
        cnn1 = DenseNet3(depth=40, num_classes=num_classes)
        cnn1.cuda()
        print(cnn1.parameters)
        optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)

        cnn2 = DenseNet3(depth=40, num_classes=num_classes)
        cnn2.cuda()
        print(cnn2.parameters)
        optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError

    epoch=0
    train_acc1=0
    train_acc2=0
    # evaluate models with random weights
    test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, args.n_epoch, len(test_loader.dataset), test_acc1, test_acc2))

    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,(test_acc1+test_acc2)/2.))
    test_log.flush()

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2)
        # evaluate models
        test_acc1, test_acc2=evaluate(test_loader, cnn1, cnn2)
        # save results
        # mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        # mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%' % (epoch+1, args.n_epoch, len(test_loader.dataset), test_acc1, test_acc2))

        test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, (test_acc1 + test_acc2) / 2.))
        test_log.flush()

    test_log.close()

if __name__=='__main__':
    main()
