import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import argparse
import json
import os
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict

# Promix Utils
def set_global_seeds(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)

def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device

class CE_Soft_Label(nn.Module):
    def __init__(self):
        super().__init__()
        # print('Calculating uniform targets...')
        # calculate confidence
        self.confidence = None
        self.gamma = 2.0
        self.alpha = 0.25
    def init_confidence(self, noisy_labels, num_class):
        noisy_labels = torch.Tensor(noisy_labels).long().cuda()
        self.confidence = F.one_hot(noisy_labels, num_class).float().clone().detach()

    def forward(self, outputs, targets=None):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * targets.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        #p = torch.exp(-loss_vec)
        #loss_vec =  (1 - p) ** self.gamma * loss_vec
        average_loss = loss_vec.mean()
        return loss_vec

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index, conf_ema_m):
        with torch.no_grad():
            _, prot_pred = temp_un_conf.max(dim=1)
            pseudo_label = F.one_hot(prot_pred, temp_un_conf.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - conf_ema_m) * pseudo_label
        return None

def linear_rampup2(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.num_epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# PGDF Utils
def pgdf_parse_args(file):
    parser = argparse.ArgumentParser(description="PyTorch PGDF CIFAR Training")
    parser.add_argument("--preset", required=True, type=str)
    parser.add_argument("--export", default='../export/pgdf', type=str)
    parser.add_argument("--noise-mode", default='sym', type=str)
    parser.add_argument("--data-path", default='../../data', type=str)
    parser.add_argument("--dataset", default='cifar10', type=str)
    parser.add_argument("--r", default=0.5, type=float)
    parser.add_argument("--noise-file", default=None)
    parser.add_argument("--log", default='eval_results', type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument('--net', type=str, default='preact_resnet18')
    parser.add_argument('--seed', type=int, default=17)
    cmdline_args = parser.parse_args()

    with open(file, "r") as f:
        jsonFile = json.load(f)

    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    args = dotdict()
    args.update(jsonFile)
    if "configs" in args:
        del args["configs"]
        jsonFile = jsonFile["configs"]

    args.preset, args.export, args.noise_mode, args.data_path, args.dataset, args.r, args.noise_file, args.log, args.gpu, args.net, args.seed\
        = cmdline_args.preset, cmdline_args.export, cmdline_args.noise_mode, cmdline_args.data_path, cmdline_args.dataset, cmdline_args.r, cmdline_args.noise_file, cmdline_args.log, cmdline_args.gpu, cmdline_args.net, cmdline_args.seed
    subpresets = cmdline_args.preset.split(".")
    for subp in subpresets:
        jsonFile = jsonFile[subp]
        args.update(jsonFile)
        if "configs" in args:
            del args["configs"]
        if "configs" in jsonFile:
            jsonFile = jsonFile["configs"]

    args.checkpoint_path = f"{args.export}/{args.dataset}/{args.net}/r{args.r}/{args.log[5:]}"
    args.pretrained_path = args.checkpoint_path

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    all_folder = os.path.join(args.checkpoint_path, "all")
    if not os.path.exists(all_folder):
        os.mkdir(all_folder)
    saved_folder = os.path.join(args.checkpoint_path, "saved")
    if not os.path.exists(saved_folder):
        os.mkdir(saved_folder)
    if not os.path.exists(args.pretrained_path + f"/saved/{args.preset}.pth.tar"):
        # if os.path.exists(args.pretrained_path + f"/saved/metrics.log"):
        #     raise AssertionError("Training log already exists!")
        args.pretrained_path = ""

    with open(args.checkpoint_path + "/saved/info.json", "w") as f:
        json.dump(args, f, indent=4, sort_keys=True)

    return args

class his_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __getitem__(self, index):
        return self.data[index],self.label[index]
    def __len__(self):
        return self.data.shape[0]

# SOP Utils
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def cosine_rampup(current, rampup_length):
    """Cosine rampup"""
    current = np.clip(current, 0.0, rampup_length)
    return float(-.5 * (np.cos(np.pi * current / rampup_length) - 1))


# T-Revision
def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error

def transition_matrix_generate(noise_rate=0.5, num_classes=10):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes-1):
            P[i, i] = 1. - n
        P[num_classes-1, num_classes-1] = 1. - n
    return P

def fit(X, num_classes, filter_outlier=False):
    # number of classes
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        if not filter_outlier:
            idx_best = np.argmax(eta_corr[:, i])
        else:
            eta_thresh = np.percentile(eta_corr[:, i], 97,interpolation='higher')
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
            idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    return T

# Meta-Learning
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

@torch.no_grad()
def update_params(params, grads, eta, opt, args, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')

# be aware that the opt state dict returns references, hence take care not to
# modify them
def update_params_sgd(params, grads, eta, opt, args, deltaonly, return_s=False):
    # supports SGD-like optimizers
    ans = []

    if return_s:
        ss = []

    wdecay = opt.defaults['weight_decay']
    momentum = opt.defaults['momentum']
    dampening = opt.defaults['dampening']
    nesterov = opt.defaults['nesterov']

    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay # s=1
        s = 1

        if momentum > 0:
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)

            moment.add_(dparam, alpha=1. -dampening) # s=1.-dampening

            if nesterov:
                dparam = dparam + momentum * moment # s= 1+momentum*(1.-dampening)
                s = 1 + momentum*(1.-dampening)
            else:
                dparam = moment # s=1.-dampening
                s = 1.-dampening

        if deltaonly:
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam  * eta)

        if return_s:
            ss.append(s*eta)

    if return_s:
        return ans, ss
    else:
        return ans

# ============== mlc step procedure debug with features (gradient-stopped) from main model ===========
#
# METANET uses the last K-1 steps from main model and imagine one additional step ahead
# to compose a pool of actual K steps from the main model
#
#
def step_hmlc_K(main_net, main_opt, hard_loss_f,
                meta_net, meta_opt, soft_loss_f,
                data_s, target_s, data_g, target_g,
                data_c, target_c,
                eta, args):
    # compute gw for updating meta_net
    logit_g = main_net(data_g)
    loss_g = hard_loss_f(logit_g, target_g)
    gw = torch.autograd.grad(loss_g, main_net.parameters())

    # given current meta net, get corrected label
    logit_s, x_s_h = main_net(data_s, return_h=True)
    pseudo_target_s = meta_net(x_s_h.detach(), target_s)
    loss_s = soft_loss_f(logit_s, pseudo_target_s)

    if data_c is not None:
        bs1 = target_s.size(0)
        bs2 = target_c.size(0)

        logit_c = main_net(data_c)
        loss_s2 = hard_loss_f(logit_c, target_c)
        loss_s = (loss_s * bs1 + loss_s2 * bs2) / (bs1 + bs2)

    f_param_grads = torch.autograd.grad(loss_s, main_net.parameters(), create_graph=True)

    f_params_new, dparam_s = update_params(main_net.parameters(), f_param_grads, eta, main_opt, args, return_s=True)
    # 2. set w as w'
    f_param = []
    for i, param in enumerate(main_net.parameters()):
        f_param.append(param.data.clone())
        param.data = f_params_new[i].data  # use data only as f_params_new has graph

    # training loss Hessian approximation
    Hw = 1  # assume to be identity

    # 3. compute d_w' L_{D}(w')
    logit_g = main_net(data_g)
    loss_g = hard_loss_f(logit_g, target_g)
    gw_prime = torch.autograd.grad(loss_g, main_net.parameters())

    # 3.5 compute discount factor gw_prime * (I-LH) * gw.t() / |gw|^2
    tmp1 = [(1 - Hw * dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
    gw_norm2 = (_concat(gw).norm()) ** 2
    tmp2 = [gw[i] / gw_norm2 for i in range(len(gw))]
    gamma = torch.dot(_concat(tmp1), _concat(tmp2))

    # because of dparam_s, need to scale up/down f_params_grads_prime for proxy_g/loss_g
    Lgw_prime = [dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]

    proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))

    # back prop on alphas
    meta_opt.zero_grad()
    proxy_g.backward()

    # accumulate discounted iterative gradient
    for i, param in enumerate(meta_net.parameters()):
        if param.grad is not None:
            param.grad.add_(gamma * args.dw_prev[i])
            args.dw_prev[i] = param.grad.clone()

    if (args.steps + 1) % (args.gradient_steps) == 0:  # T steps proceeded by main_net
        meta_opt.step()
        args.dw_prev = [0 for param in meta_net.parameters()]  # 0 to reset

    # modify to w, and then do actual update main_net
    for i, param in enumerate(main_net.parameters()):
        param.data = f_param[i]
        param.grad = f_param_grads[i].data
    main_opt.step()

    return loss_g, loss_s

class DummyScheduler(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        lrs = []
        for param_group in self.optimizer.param_groups:
            lrs.append(param_group['lr'])

        return lrs

    def step(self, epoch=None):
        pass

def tocuda(data):
    if type(data) is list:
        if len(data) == 1:
            return data[0].cuda()
        else:
            return [x.cuda() for x in data]
    else:
        return data.cuda()

def clone_parameters(model):
    assert isinstance(model, torch.nn.Module), 'Wrong model type'

    params = model.named_parameters()

    f_params_dict = {}
    f_params = []
    for idx, (name, param) in enumerate(params):
        new_param = torch.nn.Parameter(param.data.clone())
        f_params_dict[name] = idx
        f_params.append(new_param)

    return f_params, f_params_dict

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target

