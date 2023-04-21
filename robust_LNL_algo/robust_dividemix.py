from __future__ import print_function

import json
import os.path
import sys

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
# from PreResNet import *
import loader.dataloader as dataloader
import generic.adv_attack as attack

from models.PreResNet import *
from models.densenet import *
from sklearn.mixture import BayesianGaussianMixture

parser = argparse.ArgumentParser(description='PyTorch Robust DivideMix')
parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--wd', default=0.0005, type=float)
parser.add_argument('--noise-mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda-u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p-threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num-epochs', default=300, type=int)
parser.add_argument('--r', default=0.4, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=17)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num-class', default=10, type=int)
parser.add_argument('--data-path', default='../data', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noise-file', default=None)
parser.add_argument('--log', default='eval_results', type=str)
parser.add_argument('--net', type=str, default='preact_resnet18')
parser.add_argument('--lam', default=0.8, type=float)
parser.add_argument('--P', default=1.0, type=float)
parser.add_argument('--warmup', default=4, type=int)
parser.add_argument('--perturb-threshold', default=0.5, type=float)
parser.add_argument('--joint-selection', action='store_true', default=False)
parser.add_argument('--pl', default=1, type=int)
parser.add_argument('--max-iter', default=20, type=int)
parser.add_argument('--tol', default=1e-2, type=float)
parser.add_argument('--prior-type', default='dirichlet_process', type=str)
args = parser.parse_args()

if args.dataset == 'cifar10' or args.dataset == 'mnist':
    args.num_class = 10
elif args.dataset == 'cifar100':
    args.num_class = 100

if args.noise_mode == 'bad' or args.noise_mode == 'idn':
    assert args.noise_file is not None

print(args)

torch.cuda.set_device(args.gpu)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)

        penalty = conf_penalty(outputs)
        L = loss + args.P * penalty

        L.backward()  
        optimizer.step()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()
    return best_test_acc

def eval_train(model, all_loss, eval_loader, max_dist=False):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if args.r == 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component Bayesian GMM to the loss
    gmm = BayesianGaussianMixture(n_components=2, max_iter=args.max_iter, tol=args.tol, reg_covar=5e-4,
                                      weight_concentration_prior_type=args.prior_type)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    if max_dist:
        prob = prob[:, gmm.means_.argmax()]
    else:
        prob = prob[:, gmm.means_.argmin()]
    print('gmm.means: ', gmm.means_)
    return prob, all_loss

def eval_train_perturbed(model,all_loss,eval_loader,lam,max_dist=False,netid=None,epoch=None,last_prob=None):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            labels_ = attack.label_flip(model, inputs, targets, num_classes=args.num_class, step_size=lam)

            outputs = model(inputs) 
            loss = CE(outputs, labels_)
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
    losses = (losses-losses.min())/(losses.max()-losses.min())
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component Bayesian GMM to the loss
    prob_sum = np.zeros((len(input_loss),))
    for i in range(1):
        gmm = BayesianGaussianMixture(n_components=2, max_iter=args.max_iter, tol=args.tol, reg_covar=5e-4,
                                      weight_concentration_prior_type=args.prior_type)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        print('gmm%d.means: ' % i, gmm.means_)
        print('gmm%d.converged: ' % i, gmm.converged_)
        print('gmm%d.n_iter_' % i, gmm.n_iter_)
        if max_dist:
            prob = prob[:, gmm.means_.argmax()]
        else:
            prob = prob[:, gmm.means_.argmin()]
        prob_sum += prob
    prob = prob_sum / 1.
    if not gmm.converged_ and last_prob is not None:
        prob = last_prob
        print('*** BayesGMM is not converged .. Load last probability .. ***')
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def adjust_lambda(epoch, warm_up, lam, p_length=10):
    current = np.clip(1 - ((epoch - warm_up) / p_length), 0.0, 1.0)
    return lam * float(current)

def create_model():
    grayscale = True if args.dataset == 'mnist' else False
    if args.net == 'preact_resnet18':
        model = PreActResNet18(num_classes=args.num_class, grayscale=grayscale)
    elif args.net == 'densenet':
        model = DenseNet3(depth=40, num_classes=args.num_class)
    else:
        raise NotImplementedError
    model = model.cuda()
    return model

test_log = open('../eval_results/%s/%s/r%.1f/%s'%(args.dataset, args.net, args.r, args.log)+'.log', 'a')
test_log.write('===========================================\n')
test_log.write('Eval with Robust DivideMix ..\n')
test_log.flush()

warm_up = args.warmup
if args.dataset == 'cifar100' and args.noise_mode == 'bad' and args.r > 0.5:
    args.joint_selection = True

if args.dataset == 'mnist':
    loader = dataloader.mnist_dataloader(r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=5, root_dir=args.data_path, noise_file=args.noise_file)
else:
    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=5, root_dir=args.data_path, noise_file=args.noise_file)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks
p_all_loss = [[],[]]
p_labels1, p_labels2 = [], []

prob1, prob2, pred1, pred2, p_pred1, p_pred2 = None, None, None, None, None, None
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 250:
        lr /= 100
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    if epoch < warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup(net2, optimizer2, warmup_trainloader)
    else:
        lam = adjust_lambda(epoch, warm_up, args.lam, args.pl)
        print(f'===> epoch: {epoch}, lambda: {lam}.')
        if args.joint_selection and epoch == warm_up: # jointly dividing data using perturbed and un-perturbed posterior probabilities to improve data division quality
            prob1, all_loss[0] = eval_train(net1, all_loss[0], eval_loader, max_dist=True)
            prob2, all_loss[1] = eval_train(net2, all_loss[1], eval_loader, max_dist=True)
            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            p_prob1, p_all_loss[0] = eval_train_perturbed(net1, p_all_loss[0], eval_loader, lam=lam, max_dist=True,
                                                          netid=1, epoch=epoch, last_prob=prob2)
            p_prob2, p_all_loss[1] = eval_train_perturbed(net2, p_all_loss[1], eval_loader, lam=lam, max_dist=True,
                                                          netid=2, epoch=epoch, last_prob=prob1)
            p_pred1 = (p_prob1 > args.perturb_threshold)
            p_pred2 = (p_prob2 > args.perturb_threshold)
        else:
            prob_temp1, p_all_loss[0] = eval_train_perturbed(net1, p_all_loss[0], eval_loader, lam=lam, max_dist=False,
                                                             netid=1, epoch=epoch, last_prob=prob2)
            prob_temp2, p_all_loss[1] = eval_train_perturbed(net2, p_all_loss[1], eval_loader, lam=lam, max_dist=False,
                                                             netid=2, epoch=epoch, last_prob=prob1)
            prob1, prob2 = prob_temp1, prob_temp2
            pred1 = (prob1 > args.perturb_threshold)
            pred2 = (prob2 > args.perturb_threshold)
            p_pred1, p_pred2 = None, None

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2, p_pred2)  # co-divide
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1, p_pred1)  # co-divide
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

    test(epoch,net1,net2)

test_log.close()


