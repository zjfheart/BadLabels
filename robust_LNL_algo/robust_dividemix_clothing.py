from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
import loader.dataloader as dataloader
import generic.adv_attack as attack
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch-size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda-u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p-threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num-epochs', default=80, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data-path', default='../data/Clothing1M', type=str, help='path to dataset')
parser.add_argument('--seed', default=17)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num-class', default=14, type=int)
parser.add_argument('--num-batches', default=1000, type=int)
parser.add_argument('--log', default='eval_results', type=str)
parser.add_argument('--lam', default=0.2, type=float)
parser.add_argument('--warmup', default=2, type=int)
parser.add_argument('--perturb-threshold', default=0.5, type=float)
parser.add_argument('--pl', default=1, type=int)
parser.add_argument('--max-iter', default=50, type=int)
parser.add_argument('--tol', default=1e-2, type=float)
parser.add_argument('--prior-type', default='dirichlet_process', type=str)
args = parser.parse_args()

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
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step()

    
def val(net,val_loader,k,epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Validation\t Net%d  Epoch%d  Acc: %.2f%%" %(k,epoch,acc))
    test_log.write("\n| Validation\t Net%d  Epoch%d  Acc: %.2f%%" %(k,epoch,acc))
    test_log.flush()

    export_path = '../export/robdm/clothing/%s' % (args.log)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    save_point = '%s/net%d.pth.tar' % (export_path, k)
    torch.save({
        'epoch': epoch,
        'state_dict': net.state_dict()}, save_point)
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = '%s/best-net%d.pth.tar'%(export_path,k)
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict()}, save_point)
    return acc

def test(net1,net2,test_loader):
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
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))
    test_log.write('| Test Accuracy:%.2f\n'%(acc))
    test_log.flush()
    return acc    
    
def eval_train_perturbed(eval_loader, model, lam, max_dist=False, last_prob=None):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            labels_ = attack.label_flip(model, inputs, targets, num_classes=args.num_class, step_size=lam, num_steps=1)

            outputs = model(inputs) 
            loss = CE(outputs, labels_)
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                paths.append(path[b])
                n+=1
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)

    # fit a BayesGMM to the loss
    prob_sum = np.zeros((len(losses),))
    for i in range(1):
        gmm = BayesianGaussianMixture(n_components=2, max_iter=args.max_iter, tol=args.tol, reg_covar=5e-4,
                                      weight_concentration_prior_type=args.prior_type)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        print('gmm%d.means: ' % i, gmm.means_)
        print('gmm%d.converged: ' % i, gmm.converged_)
        print('gmm%d.n_iter_' % i, gmm.n_iter_)
        if max_dist:
            prob = prob[:, gmm.means_.argmax()]
        else:
            prob = prob[:, gmm.means_.argmin()]
        # print(prob.shape)
        prob_sum += prob
        # print(prob_sum)
    prob = prob_sum / 1.
    # print(prob)
    if not gmm.converged_ and last_prob is not None:
        prob = last_prob
        print('*** BayesGMM is not converged .. Load last probability .. ***')

    return prob,paths
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def adjust_lambda(epoch, warm_up, lam, p_length=1):
    current = np.clip(1 - ((epoch - warm_up) / p_length), 0.0, 1.0)
    return lam * float(current)
               
def create_model():
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048,args.num_class)
    model = model.cuda()
    return model     

test_log = open('../eval_results/clothing/%s'%(args.log)+'.log', 'a')
test_log.write('===========================================\n')
test_log.write('Eval with Robust DivideMix on Clothing 1M ..\n')
test_log.flush()

warm_up = args.warmup

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                      
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

best_acc = [0,0]
prob1, prob2 = None, None
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10       
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr     
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr    
        
    if epoch<warm_up:     # warm up
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2,optimizer2,train_loader)
    else:       
        pred1 = (prob1 > args.perturb_threshold)  # divide dataset
        pred2 = (prob2 > args.perturb_threshold)
        
        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)              # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)              # train net2
    
    val_loader = loader.run('val') # validation
    acc1 = val(net1,val_loader,1,epoch)
    acc2 = val(net2,val_loader,2,epoch)

    if epoch >= warm_up-1:
        lam = adjust_lambda(epoch, warm_up-1, args.lam, args.pl)
        print(f'===> epoch: {epoch}, lambda: {lam}.')
        eval_loader = loader.run('eval_train')
        prob_temp1, paths1 = eval_train_perturbed(eval_loader, net1, lam=lam, max_dist=False, last_prob=prob2)
        eval_loader = loader.run('eval_train')
        prob_temp2, paths2 = eval_train_perturbed(eval_loader, net2, lam=lam, max_dist=False, last_prob=prob1)
        prob1, prob2 = prob_temp1, prob_temp2

test_loader = loader.run('test')
net1.load_state_dict(torch.load('../export/robdm/clothing/%s/best-net1.pth.tar'%(args.log))['state_dict'])
net2.load_state_dict(torch.load('../export/robdm/clothing/%s/best-net2.pth.tar'%(args.log))['state_dict'])
acc = test(net1,net2,test_loader)      

