import argparse
import os
import numpy as np
import json
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

from utils import flip_labels, compute_z_gradient, softmax
from datasets import IndexCIFAR10, IndexCIFAR100, IndexSVHN, IndexMNIST
from models.PreResNet import *

parser = argparse.ArgumentParser(description='Pytorch Generate BadLabels')

parser.add_argument('--arch', type=str, default='preact_resnet18')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--data-path', type=str, default='../data')
parser.add_argument('--pretrain', type=str, default=None)

parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--flip-budget', type=float, default=0.2)
parser.add_argument('--attack-iter', type=int, default=120)
parser.add_argument('--step-size', type=float, default=0.1)

parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--noise-file', type=str, default='../eval_badlabels/noise')
parser.add_argument('--z-path', type=str, default='./z')
parser.add_argument('--z-load', type=str, default=None)
parser.add_argument('--gen', action='store_true', default=False)

args = parser.parse_args()
print('args', args)

torch.cuda.set_device(args.gpu)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if not os.path.exists(args.noise_file):
    os.makedirs(args.noise_file)
if not os.path.exists(args.z_path):
    os.makedirs(args.z_path)

"""load dataset"""
if args.dataset == 'mnist':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
elif args.dataset == 'svhn':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
elif args.dataset == 'cifar10':
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
else:
    raise NotImplementedError

if args.dataset == 'mnist':
    trainset = IndexMNIST(args.data_path, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = IndexMNIST(args.data_path, train=False, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 10
    grayscale = True
elif args.dataset == 'svhn':
    trainset = IndexSVHN(root=args.data_path, split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = IndexSVHN(root=args.data_path, split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 10
    grayscale = False
elif args.dataset == 'cifar10':
    trainset = IndexCIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = IndexCIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 10
    grayscale = False
elif args.dataset == "cifar100":
    trainset = IndexCIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = IndexCIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    num_classes = 100
    grayscale = False
else:
    raise NotImplementedError

"""load model"""
if args.arch == 'preact_resnet18':
    model = PreActResNet18(num_classes=num_classes, grayscale=grayscale).cuda()
else:
    raise NotImplementedError

"""init"""
print('='*30)
num_samples = len(trainset)
print('num of labels: ', num_samples, '\n')
shape = (num_samples, num_classes)
# z = softmax(np.random.uniform(size=(num_labels, num_classes)))
z = F.one_hot(torch.tensor(trainset.targets), num_classes).numpy()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

"""load z OR training from scratch"""
if args.z_load is not None:
    print(f'load z from {args.z_load} ...')
    z = np.load(args.z_load)
else:
    for epoch in range(args.epochs):
        losses = []
        z_grad = np.zeros(shape)
        print('z grad shape', z_grad.shape)

        correct = 0
        model.train()
        for batch_idx, (data, target, index) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss = nn.CrossEntropyLoss(reduction='none')(output, target)
            losses.append(loss)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            z_grad_ = compute_z_gradient(output)
            for i in range(len(index)):
                z_grad[index[i]] = z_grad_[i].detach().cpu().numpy()

        scheduler.step()
        train_acc = correct / num_samples

        z = z - args.step_size * z_grad

        correct = 0
        model.eval()
        with torch.no_grad():
            for data, target, _ in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_acc = correct / len(test_loader.dataset)

        print(f'Epoch {epoch + 1} || Train acc: {train_acc:2.2f} || Test acc: {test_acc:2.2f}')

        if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs:
            z_path = os.path.join(args.z_path, f'z_{args.dataset}_e{epoch + 1}.npy')
            print(f'save z to {z_path} ...')
            np.save(z_path, z)

"""gen"""
if args.gen:
    z = softmax(z)
    y_flipped = flip_labels(z, trainset, args.flip_budget)
    noise_labels = y_flipped.clone().numpy().tolist()
    noise_name = 'badlabels_' + args.dataset + '_r' + str(args.flip_budget) + '.json'
    noise_file = os.path.join(args.noise_file, noise_name)
    print(f"save noisy labels to {noise_file} ...")
    json.dump(noise_labels, open(noise_file, "w"))


