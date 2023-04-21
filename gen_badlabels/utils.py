import os
import random

import numpy as np
import torch
from torch import nn

def flip_labels(z, trainset, budget):
    indices = np.argsort(z, axis=1)
    min = indices[:, 0]
    sec = indices[:, 1]
    flip_to_idx = []
    labels_ = torch.tensor(trainset.targets).clone()
    for i in range(len(z)):
        if labels_[i] == min[i]:
            flip_to_idx.append(sec[i])
        else:
            flip_to_idx.append(min[i])
    flip_to_zvalue = [z[i][flip_to_idx[i]] for i in range(len(z))]
    noise_label_idx = np.argsort(flip_to_zvalue)
    num_flipped = 0
    budget = int(len(trainset) * budget)
    while num_flipped < budget:
        ori = int(labels_[noise_label_idx[num_flipped]])
        m = int(min[noise_label_idx[num_flipped]])
        s = int(sec[noise_label_idx[num_flipped]])
        labels_[noise_label_idx[num_flipped]] = flip_to_idx[noise_label_idx[num_flipped]]
        print("ori: %d, min: %d, sec: %d, flip-to: %d" % (ori, m, s, flip_to_idx[noise_label_idx[num_flipped]]))
        num_flipped += 1
    return labels_

def compute_z_gradient(output):
    return -torch.log(torch.softmax(output, dim=1))

def compute_loss_penalty(losses, shape):
    mean_loss = losses.mean()
    gap = torch.abs(losses - mean_loss).unsqueeze(1).detach().cpu()
    penalty = torch.ones(size=shape) * gap
    # print('penalty', penalty)
    return penalty.numpy()

def softmax(logits):
    e_x = np.exp(logits)
    probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
    return probs

def save_checkpoint(state, path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(path, filename)
    torch.save(state, filepath)

