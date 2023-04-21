import argparse
import random

import sys

import loader.dataloader as dataloader
sys.path.append("..")
from models.small_cnn import *
from models.resnet import *
from models.PreResNet import *
from models.densenet import *

parser = argparse.ArgumentParser(description='Pytorch Standard Training')

parser.add_argument('--net', type=str, default='preact_resnet18')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--noise-mode',  default='sym')
parser.add_argument('--batch-size', default=128, type=int, help='train batchsize')
parser.add_argument('--noise-file', default=None)
parser.add_argument('--num-classes', default=10)
parser.add_argument('--seed', type=int, default=17)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--log', type=str, default='eval_results')
parser.add_argument('--data-path', type=str, default='../../data')

args = parser.parse_args()

if args.dataset == 'cifar10' or args.dataset == 'mnist':
    args.num_classes = 10
elif args.dataset == 'cifar100':
    args.num_classes = 100

print(args)

torch.cuda.set_device(args.gpu)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

"""load dataset"""
test_log = open('../eval_results/%s/%s/r%.1f/%s'%(args.dataset, args.net, args.r, args.log)+'.log', 'a')
test_log.write('===========================================\n')
test_log.write('Eval with Standard Training ..\n')
test_log.flush()
if args.dataset == 'mnist':
    loader = dataloader.mnist_dataloader(r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=5, root_dir=args.data_path, noise_file=args.noise_file)
else:
    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=5,
                                         root_dir=args.data_path, noise_file=args.noise_file)
train_loader = loader.run('warmup')
test_loader = loader.run('test')


"""load model"""
grayscale = True if args.dataset == 'mnist' else False
if args.net == 'preact_resnet18':
    model = PreActResNet18(num_classes=args.num_classes, grayscale=grayscale).cuda()
elif args.net == 'densenet':
    model = DenseNet3(depth=40, num_classes=args.num_classes).cuda()
else:
    raise NotImplementedError

"""training"""
print('='*15, 'training', '='*15)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)

start_epoch = 0
for epoch in range(start_epoch, args.epochs):
    print('epoch ', epoch + 1)
    loss_sum = 0
    for batch_idx, (data, target, path) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss(reduction='mean')(output, target)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    train_loss = loss_sum / len(train_loader.dataset)
    print("train loss: ", train_loss)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset) * 100.
    print("test loss: ", test_loss)
    print("test accuracy: ", test_accuracy)
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,test_accuracy))
    test_log.flush()

test_log.close()
