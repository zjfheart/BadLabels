import torch.nn as nn
import sys
import torch.nn.functional as F
sys.path.append("..")
from models.resnet import ResNet, BasicBlock
from models.PreResNet import PreActResNet, PreActBlock
from models.parameterization_net import LabelParameterization
from models.densenet import DenseNet3

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def reparameterization(n_samples = 50000, num_classes = 10, init='gaussian', mean=0., std=1e-4):
	return LabelParameterization(n_samples = n_samples, n_class = num_classes, init=init, mean=mean, std=std)


def resnet50(num_classes=14, rotation = False):
    import torchvision.models as models
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

def densenet(num_classes=10):
    return DenseNet3(40, num_classes=num_classes, growth_rate=12)

def PreActResNet34(num_classes=10) -> PreActResNet:
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes)
def preact_resnet18(num_classes=10, grayscale=False) -> PreActResNet:
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, grayscale=grayscale)