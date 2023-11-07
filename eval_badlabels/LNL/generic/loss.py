from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .sop_parse_config import ConfigParser
from numpy.testing import assert_array_almost_equal

# Co-teaching Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_2_update=ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    # pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]])/float(num_remember)
    # pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_update]])/float(num_remember)

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, step):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id=np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    temp_disagree = ind*logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]
     
    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id] 
        update_outputs2 = outputs2[disagree_id] 
        
        loss_1, loss_2 = loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate, ind_disagree)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step*cross_entropy_1)/labels.size()[0]
        loss_2 = torch.sum(update_step*cross_entropy_2)/labels.size()[0]
 
        # pure_ratio_1 = np.sum(noise_or_not[ind])/ind.shape[0]
        # pure_ratio_2 = np.sum(noise_or_not[ind])/ind.shape[0]
    return loss_1, loss_2


# SOP Loss functions
cross_entropy_val = nn.CrossEntropyLoss

class cross_entropy(nn.Module):
    def __init__(self, num_examp=50000, num_classes=10, alpha=0.3, ratio_consistency=0, ratio_balance=0, ratio_reg=100):
        super(cross_entropy, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.s = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.t = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.E = None

    def compute_loss(self):
        param_y = self.s ** 2 - self.t ** 2
        max_, _ = torch.max(param_y, dim=1)
        return torch.mean(max_)

    def forward(self, index, outputs, label):
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
        else:
            output = outputs

        ce_loss = F.cross_entropy(output, label.argmax(dim=1))
        return ce_loss, None, 0

class overparametrization_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, ratio_consistency=0, ratio_balance=0):
        super(overparametrization_loss, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp

        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

        self.init_param(mean=self.config['reparam_arch']['args']['mean'],
                        std=self.config['reparam_arch']['args']['std'])

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, outputs, label):
        # label = torch.zeros(len(label), self.config['num_classes']).cuda().scatter_(1, label.view(-1,1), 1)

        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)

            ensembled_output = 0.5 * (output + output2).detach()

        else:
            output = outputs

            ensembled_output = output.detach()

        eps = 1e-4

        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)

        E = U_square - V_square

        self.E = E

        original_prediction = F.softmax(output, dim=1)

        prediction = torch.clamp(original_prediction + U_square - V_square.detach(), min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)

        prediction = torch.clamp(prediction, min=eps, max=1.0)

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = F.mse_loss((label_one_hot + U_square - V_square), label, reduction='sum') / len(label)

        loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim=-1))

        loss += MSE_loss

        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

            loss += self.ratio_balance * balance_kl

        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):
            consistency_loss = self.consistency_loss(index, output, output2)

            loss += self.ratio_consistency * torch.mean(consistency_loss)

        return loss

    def consistency_loss(self, index, output1, output2):
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.config['num_classes'])).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1),
                                                                                     1)


# ELR Loss functions
def cross_entropy_elr_val(output, target):
    return F.cross_entropy(output, target)


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp,
                                                                                                        self.num_classes)
        self.beta = beta

    def forward(self, index, output, label):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
                    (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.config['train_loss']['args']['lambda'] * elr_reg
        return final_loss


# Negative Label Smoothing
def loss_gls(epoch, y, t, smooth_rate=0.1, wa=0, wb=1):
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(y, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=t.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = (wa + wb * confidence) * nll_loss + wb * smooth_rate * smooth_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    return torch.sum(loss)/num_batch


# T-Revision
class reweight_loss(nn.Module):
    def __init__(self):
        super(reweight_loss, self).__init__()

    def forward(self, out, T, target):
        loss = 0.
        out_softmax = F.softmax(out, dim=1)
        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0)
            temp_softmax = torch.unsqueeze(temp_softmax, 0)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]]
            out_T = torch.matmul(T.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]]
            beta = pro1 / pro2
            beta = Variable(beta, requires_grad=True)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)


class reweighting_revision_loss(nn.Module):
    def __init__(self):
        super(reweighting_revision_loss, self).__init__()

    def forward(self, out, T, correction, target):
        loss = 0.
        out_softmax = F.softmax(out, dim=1)
        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0)
            temp_softmax = torch.unsqueeze(temp_softmax, 0)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]]
            T = T + correction
            T_result = T
            out_T = torch.matmul(T_result.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]]
            beta = (pro1 / pro2)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)

# Meta-Learning
def soft_cross_entropy(logit, pseudo_target, reduction='mean'):
    loss = -(pseudo_target * F.log_softmax(logit, -1)).sum(-1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementedError('Invalid reduction: %s' % reduction)
