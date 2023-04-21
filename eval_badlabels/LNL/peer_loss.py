"""
Original Code is from https://github.com/weijiaheng/Multi-class-Peer-Loss-functions
"""

import random
import sys
import numpy as np
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
import torch
import argparse
torch.autograd.set_detect_anomaly(True)
import loader.dataloader as dataloader

sys.path.append("..")
from models.PreResNet import *
from models.densenet import *


parser = argparse.ArgumentParser(description="Pytorch Peer Loss")
parser.add_argument('--r', type=float, required=True, help='category of noise label')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--bias', type=str, required=False)

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--noise-mode',  default='sym')
parser.add_argument('--noise-file', default=None)
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log', type=str, default='eval_results')
parser.add_argument('--data-path', type=str, default='../../data')
parser.add_argument('--net', type=str, default='preact_resnet18')
opt = parser.parse_args()

if opt.dataset == 'cifar10' or opt.dataset == 'mnist':
    num_classes = 10
elif opt.dataset == 'cifar100':
    num_classes = 100

root = opt.data_path
r = opt.r
batch_size = opt.batchsize
num_epochs = 300

CUDA = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")

torch.cuda.set_device(opt.gpu)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

# Stable CE
class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )

        
criterion = CrossEntropyLossStable()
criterion.to(device)


# Training
def train(train_loader, peer_loader, model, optimizer, epoch):

    model.train()
    for i, (input, target, idx) in enumerate(train_loader):
        if idx.size(0) != batch_size*2:
            continue
        input = torch.autograd.Variable(input.to(device))
        target = torch.Tensor(target.float())
        target = torch.autograd.Variable(target.to(device))
        output = model(input)
        optimizer.zero_grad()
        
        # Prepare mixmatched images and labels for the Peer Term
        peer_iter = iter(peer_loader)
        input1 = peer_iter.next()[0]
        output1 = model(input1.to(device))
        target2 = peer_iter.next()[1]
        target2 = torch.Tensor(target2.float())
        target2 = torch.autograd.Variable(target2.to(device))
        # Peer Loss with Cross-Entropy loss: L(f(x), y) - L(f(x1), y2)
        loss = criterion(output, target.long()) - f_alpha(epoch) * criterion(output1, target2.long())
        loss.to(device)
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
            
    for i, (input, target) in enumerate(test_loader):
        input = torch.Tensor(input).to(device)
        target = torch.Tensor(target.float()).to(device)
        total += target.size(0)
        output = model(input)
        _, predicted = torch.max(output.detach(), 1)
        correct += predicted.eq(target.long()).sum().item()
    
    accuracy = 100. * correct / total

    return accuracy


# The weight of peer term
def f_alpha(epoch):
    if opt.r == 0.1 or opt.r == 0.2:
    # Sparse setting
        alpha1 = np.linspace(0.0, 0.0, num=20)
        alpha2 = np.linspace(0.0, 1, num=20)
        alpha3 = np.linspace(1, 2, num=50)
        alpha4 = np.linspace(2, 5, num=50)
        alpha5 = np.linspace(5, 10, num=100)
        alpha6 = np.linspace(10, 20, num=100)
    else:
    # Uniform/Random noise setting
        alpha1 = np.linspace(0.0, 0.0, num=20)
        alpha2 = np.linspace(0.0, 0.1, num=20)
        alpha3 = np.linspace(1, 2, num=50)
        alpha4 = np.linspace(2, 2.5, num=50)
        alpha5 = np.linspace(2.5, 3.3, num=100)
        alpha6 = np.linspace(3.3, 5, num=100)
     
    alpha = np.concatenate((alpha1, alpha2, alpha3, alpha4, alpha5, alpha6),axis=0)
    return alpha[epoch]
   
# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch, lr_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]/(1+f_alpha(epoch))
        
        
def main():
    grayscale = True if opt.dataset == 'mnist' else False
    if opt.net == 'preact_resnet18':
        model_PL = PreActResNet18(num_classes=num_classes, grayscale=grayscale).to(device)
    elif opt.net == 'densenet':
        model_PL = DenseNet3(depth=40, num_classes=num_classes).to(device)
    else:
        raise NotImplementedError
    best_val_acc = 0
    train_acc_result = []
    val_acc_noisy_result = []
    test_acc_result = []
    
    # Hyper-parameters: learning rate and the weight of peer term \alpha
    lr_list = [0.1] * 40 + [0.01] * 40 + [0.001] * 40 + [1e-4] * 40 + [1e-5] * 40 + [1e-6] * 40 +  [1e-7] * 40 +  [1e-8] * 20

    if opt.dataset == 'mnist':
        loader = dataloader.mnist_dataloader(r=opt.r, noise_mode=opt.noise_mode, batch_size=batch_size,
                                             num_workers=0, root_dir=opt.data_path, noise_file=opt.noise_file)
    else:
        loader = dataloader.cifar_dataloader(opt.dataset, r=opt.r, noise_mode=opt.noise_mode, batch_size=batch_size,
                                             num_workers=1,
                                             root_dir=opt.data_path, noise_file=opt.noise_file)


    peer_train = loader.run('peer')

    torch.manual_seed(opt.seed + 1)
    train_loader = loader.run('warmup')
    test_loader = loader.run('test')

    for epoch in range(num_epochs):
        print("epoch=", epoch,'r=', opt.r)
        learning_rate = lr_list[epoch]

        # We adopted the SGD optimizer
        optimizer_PL = torch.optim.SGD(model_PL.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)
        # asjust the learning rate
        adjust_learning_rate(optimizer_PL, epoch, lr_list)
        train(train_loader=train_loader, peer_loader=peer_train, model=model_PL, optimizer=optimizer_PL, epoch=epoch)
        print("validating model_PL...")

        
        # Calculate test accuracy
        test_acc = test(model=model_PL, test_loader=test_loader)
        test_acc_result.append(test_acc)
        print('test_acc=', test_acc)

        test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, test_acc))
        test_log.flush()
    

if __name__ == '__main__':
    
    # Save statistics
    print("Begin:")
    test_log = open('../eval_results/%s/%s/r%.1f/%s' % (opt.dataset, opt.net, opt.r, opt.log) + '.log', 'a')
    test_log.write('===========================================\n')
    test_log.write('Eval with Peer Loss ..\n')
    test_log.flush()
            
    main()
    # evaluate('./trained_models/' + str(opt.r) + '_' + str(opt.seed))
    print("Traning finished")
    test_log.close()
