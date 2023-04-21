"""
Original Code is from https://github.com/Justherozen/ProMix
"""

from __future__ import print_function
import sys
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import loader.promix_dataloader as dataloader
from generic.utils import linear_rampup2, CE_Soft_Label, adjust_learning_rate
from augment.fmix import *
sys.path.append("..")
from models.dualnet import *

parser = argparse.ArgumentParser(description='PyTorch ProMix')
parser.add_argument('--batch-size', default=256, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, help='initial learning rate')
parser.add_argument('-lr-decay-rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', default=True,
                    help='use cosine lr schedule')
parser.add_argument('--noise-type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100',
                    default='sym')
# parser.add_argument('--noise-path', type=str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--proto-m', default=0.9, type=float, help='speed of prototype updating')
parser.add_argument('--p-threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num-epochs', default=420, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=17)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num-class', default=10, type=int)
parser.add_argument('--data-path', default='../../data', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
# parser.add_argument('--is-human', action='store_true', default=False)
parser.add_argument('--rho-range', default='0.5,0.5', type=str,
                    help='ratio of clean labels (rho)')
parser.add_argument('--tau', default=0.99, type=float,
                    help='high-confidence selection threshold')
parser.add_argument('--pretrain-ep', default=10, type=int)
parser.add_argument('--warmup-ep', default=50, type=int)
parser.add_argument('--topk', default=4, type=int)
parser.add_argument('--unrel-pseudo', default='sharpen', type=str)
parser.add_argument('--low-conf-del', action='store_true', default=False)
parser.add_argument('--threshold', default=0.9, type=float)
parser.add_argument('--fmix', action='store_true', default=False)
parser.add_argument('--use-unrel', action='store_true', default=False)
parser.add_argument('--start-expand', default=150, type=int)
parser.add_argument('--save-note', type=str,  default='')

parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--noise-file', default=None)
parser.add_argument('--log', default='eval_results', type=str)
parser.add_argument('--net', type=str, default='preact_resnet18')

args = parser.parse_args()
[args.rho_start, args.rho_end] = [float(item) for item in args.rho_range.split(',')]
if args.dataset == 'cifar10' or args.dataset == 'mnist':
    args.num_class = 10
    num_samples = 50000 if args.dataset == 'cifar10' else 60000
elif args.dataset == 'cifar100':
    args.num_class = 100
    args.pretrain_ep = 30
    args.tau = 0.95
    num_samples = 50000
print(args)

torch.cuda.set_device(args.gpu)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def high_conf_sel2(idx_chosen, w_x, batch_size, score1, score2, match):
    w_x2 = w_x.clone()
    if (1. * idx_chosen.shape[0] / batch_size) < args.threshold:
        # when clean data is insufficient, try to incorporate more examples
        high_conf_cond2 = (score1 > args.tau) * (score2 > args.tau) * match
        # both nets agrees
        high_conf_cond2 = (1. * high_conf_cond2 - w_x.squeeze()) > 0
        # remove already selected examples; newly selected
        hc2_idx = torch.where(high_conf_cond2)[0]

        max_to_sel_num = int(batch_size * args.threshold) - idx_chosen.shape[0]
        # maximally select batch_size * args.threshold; idx_chosen.shape[0] select already
        if high_conf_cond2.sum() > max_to_sel_num:
            # to many examples selected, remove some low conf examples
            score_mean = (score1 + score2) / 2
            idx_remove = (-score_mean[hc2_idx]).sort()[1][max_to_sel_num:]
            # take top scores
            high_conf_cond2[hc2_idx[idx_remove]] = False
        w_x2[high_conf_cond2] = 1
    return w_x2

# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.train()  # fix one network and train the other
    
    rho = args.rho_start + (args.rho_end - args.rho_start) * linear_rampup2(epoch, args.warmup_ep)
    w = linear_rampup2(epoch, args.warmup_ep)
    beta = 0.1

    #unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x, w_x2, true_labels, index) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)
        w_x2 = w_x2.view(-1, 1).type(torch.FloatTensor)

        index = index.cuda()
        inputs_x, inputs_x2, labels_x, w_x , w_x2= inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda(), w_x2.cuda()
        outputs_x = net(inputs_x)
        outputs_x2 = net(inputs_x2)
        outputs_a = net2(inputs_x)
        outputs_a2 = net2(inputs_x2)
        
        with torch.no_grad():
            # label refinement of labeled samples
            px = torch.softmax(outputs_x, dim=1)
            px2 = torch.softmax(outputs_a, dim=1)
            pred_net = F.one_hot(px.max(dim=1)[1], args.num_class).float()
            pred_net2 = F.one_hot(px2.max(dim=1)[1], args.num_class).float()

            high_conf_cond = (labels_x * px).sum(dim=1) > args.tau
            high_conf_cond2 = (labels_x * px2).sum(dim=1) > args.tau
            w_x[high_conf_cond] = 1
            w_x2[high_conf_cond2] = 1
            pseudo_label_l = labels_x * w_x + pred_net * (1 - w_x)
            pseudo_label_l2 = labels_x * w_x2 + pred_net2 * (1 - w_x2)

            idx_chosen = torch.where(w_x == 1)[0]
            idx_chosen_2 = torch.where(w_x2 == 1)[0]
            # selected examples

            if epoch > args.num_epochs - args.start_expand:
                # only add these points at the last 100 epochs
                score1 = px.max(dim=1)[0]
                score2 = px2.max(dim=1)[0]
                match = px.max(dim=1)[1] == px2.max(dim=1)[1]

                hc2_sel_wx1 = high_conf_sel2(idx_chosen, w_x, batch_size, score1, score2, match)
                hc2_sel_wx2 = high_conf_sel2(idx_chosen_2, w_x2, batch_size, score1, score2, match)

                idx_chosen = torch.where(hc2_sel_wx1 == 1)[0]
                idx_chosen_2 = torch.where(hc2_sel_wx2 == 1)[0]
            # Label Guessing

        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        X_w_c = inputs_x[idx_chosen]
        pseudo_label_c = pseudo_label_l[idx_chosen]
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
        pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        logits_mix = net(X_w_c_mix)
        loss_mix = CEsoft(logits_mix, targets=pseudo_label_c_mix).mean()
        # mixup loss
        x_fmix = fmix(X_w_c)
        logits_fmix = net(x_fmix)
        loss_fmix = fmix.loss(logits_fmix, (pseudo_label_c.detach()).long())
        # fmixup loss
        loss_cr = CEsoft(outputs_x2[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
        # consistency loss
        loss_ce = CEsoft(outputs_x[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
        # above: loss for reliable samples

        loss_net1 = loss_ce + w * (loss_cr + loss_mix + loss_fmix)
        #  -------  loss for net1

        l = np.random.beta(4, 4)
        l = max(l, 1-l)
        X_w_c = inputs_x[idx_chosen_2]
        pseudo_label_c = pseudo_label_l2[idx_chosen_2]
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix2 = l * X_w_c + (1 - l) * X_w_c_rand        
        pseudo_label_c_mix2 = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        logits_mix2 = net2(X_w_c_mix2)
        loss_mix2 = CEsoft(logits_mix2, targets=pseudo_label_c_mix2).mean()
        # mixup loss
        x_fmix2 = fmix(X_w_c)
        logits_fmix2 = net2(x_fmix2)
        loss_fmix2 = fmix.loss(logits_fmix2, (pseudo_label_c.detach()).long())
        # fmixup loss
        loss_cr2 = CEsoft(outputs_a2[idx_chosen_2], targets=pseudo_label_l2[idx_chosen_2]).mean()
        # consistency loss
        loss_ce2 = CEsoft(outputs_a[idx_chosen_2], targets=pseudo_label_l2[idx_chosen_2]).mean()
        loss_net2 = loss_ce2 + w * (loss_cr2 + loss_mix2 + loss_fmix2)
        #  -------  loss for net2

        loss = loss_net1 + loss_net2
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0 :
            print('%s:%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Net1 loss: %.2f  Net2 loss: %.2f'
                         % (args.dataset, args.noise_type, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss_net1.item(), loss_net2.item()))


def warmup(epoch, net, net2, optimizer, dataloader):
    net.train()
    net2.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs_w, inputs_s, labels, _) in enumerate(dataloader):
        inputs_w, inputs_s, labels = inputs_w.cuda(), inputs_s.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs_w)
        outputs2 = net2(inputs_w)
        l_ce = CEloss(outputs, labels)
        l_ce2 = CEloss(outputs2, labels)

        loss = l_ce + l_ce2

        if(loss.isnan().any()):
            print("nan in loss")
        penalty = conf_penalty(outputs)
        if(penalty.isnan().any()):
            print("nan in penalty")
        L = loss + penalty
        L.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('%s:%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f  Penalty-loss: %.4f  All-loss: %.4f'
                         % (
                         args.dataset, args.noise_type, epoch, args.num_epochs, batch_idx + 1, num_iter,loss.item(),penalty.item(), L.item()))

def evaluate(loader, model, save = False, best_acc = 0.0):
    model.eval()    # Change model to 'eval' mode.
    
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images = torch.autograd.Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc


def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    correct2 = 0
    correctmean = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)

            score1, predicted = torch.max(outputs1, 1)
            score2, predicted_2 = torch.max(outputs2, 1)
            outputs_mean = (outputs1 + outputs2) / 2
            _, predicted_mean = torch.max(outputs_mean, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            correct2 += predicted_2.eq(targets).cpu().sum().item()
            correctmean += predicted_mean.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    acc2 = 100. * correct2 / total
    accmean = 100. * correctmean / total
    print("| Test Epoch #%d\t Acc Net1: %.2f%%, Acc Net2: %.2f%% Acc Mean: %.2f%%\n" % (epoch, acc, acc2, accmean))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, accmean))
    test_log.flush()


def eval_train(model, all_loss, rho, num_class):
    model.eval()
    losses = torch.zeros(num_samples)
    targets_list = torch.zeros(num_samples)
    num_class = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            num_class = outputs.shape[1]
            loss = CE(outputs, targets)
            targets_cpu = targets.cpu()
            pred = torch.softmax(outputs, dim=1).cpu()
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                targets_list[index[b]] = targets_cpu[b]

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    input_loss = losses.reshape(-1, 1)

    prob = np.zeros(targets_list.shape[0])
    idx_chosen_sm = []
    min_len = 1e10
    for j in range(num_class):
        indices = np.where(targets_list.cpu().numpy()==j)[0]
        # torch.where will cause device error
        if len(indices) == 0:
            continue
        bs_j = targets_list.shape[0] * (1. / num_class)
        pseudo_loss_vec_j = losses[indices]
        sorted_idx_j = pseudo_loss_vec_j.sort()[1].cpu().numpy()
        partition_j = max(min(int(math.ceil(bs_j*rho)), len(indices)), 1)
        # at least one example
        idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])
        min_len = min(min_len, partition_j)

    idx_chosen_sm = np.concatenate(idx_chosen_sm)
    prob[idx_chosen_sm] = 1

    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        outputs = outputs.clamp(min=1e-12)
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    grayscale = True if args.dataset == 'mnist' else False
    model = DualNet(args.net, args.num_class, grayscale)
    model = model.cuda()
    return model


# stats_log = open('./checkpoint/%s_%s_%s' % (args.dataset, args.noise_type, args.num_epochs) + '_stats.txt', 'w')
test_log = open('../eval_results/%s/%s/r%.1f/%s'%(args.dataset, args.net, args.r, args.log)+'.log', 'a')
test_log.write('===========================================\n')
test_log.write('Eval with ProMix ..\n')
test_log.flush()

warm_up = args.pretrain_ep

if args.dataset == 'mnist':
    loader = dataloader.promix_mnist_dataloader(r=args.r, noise_type=args.noise_type, batch_size=args.batch_size, num_workers=5,
                                                root_dir=args.data_path, noise_file=args.noise_file)
    size = (28, 28)
else:
    loader = dataloader.promix_cifar_dataloader(args.dataset, r=args.r, noise_type=args.noise_type,
                                                batch_size=args.batch_size,
                                                num_workers=5, root_dir=args.data_path, noise_file=args.noise_file)
    size = (32, 32)

print('| Building net')
dualnet = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
conf_penalty = NegEntropy()
optimizer1 = optim.SGD([{'params': dualnet.net1.parameters()},
                        {'params': dualnet.net2.parameters()}
                        ], lr=args.lr, momentum=0.9, weight_decay=5e-4)

fmix = FMix(size=size)
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
CEsoft = CE_Soft_Label()

labeled_trainloader = None
unlabeled_trainloader = None
eval_loader = None
# idx2label = (torch.load(args.noise_path))[args.noise_type].reshape(-1)
eval_loader = loader.run('eval_train')
test_loader = loader.run('test')

all_loss = [[], []]  # save the history of losses from two networks

best_acc = 0
for epoch in range(args.num_epochs + 1):
    adjust_learning_rate(args, optimizer1, epoch)

    if epoch < warm_up:
        warmup_trainloader = loader.run('warmup')

        print('Warmup Net1')
        warmup(epoch, dualnet.net1, dualnet.net2, optimizer1, warmup_trainloader)

    else:
        rho = args.rho_start + (args.rho_end - args.rho_start) * linear_rampup2(epoch, args.warmup_ep)
        prob1, all_loss[0] = eval_train(dualnet.net1, all_loss[0], rho, args.num_class)
        prob2, all_loss[0] = eval_train(dualnet.net2, all_loss[0], rho, args.num_class)
        pred1 = (prob1 > args.p_threshold)
        # print('Train Net1')
        total_trainloader = loader.run('train', pred1, prob1, prob2)  # co-divide
        train(epoch,dualnet.net1, dualnet.net2, optimizer1, total_trainloader, unlabeled_trainloader) 
    
    test(epoch, dualnet.net1, dualnet.net2)

test_log.close()