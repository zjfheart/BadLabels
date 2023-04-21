import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def label_flip(model, data, target, num_classes, step_size, num_steps=1, rand_init=False):
    model.eval()

    if rand_init:
        soft_label = torch.tensor(np.random.uniform(size=(len(target), num_classes))).cuda()
    else:
        soft_label = F.one_hot(target, num_classes)

    output = model(data)
    model.zero_grad()

    # print("")
    target_ = target
    for k in range(num_steps):
        grad = -torch.log(torch.softmax(output, dim=1).add(1e-10))
        # print('grad', grad[1:5])
        soft_label = soft_label + step_size * grad
        target_ = torch.argmax(soft_label, dim=1)

    return target_
