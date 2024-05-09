import torch
from torch import nn
import torch.nn.functional as F

import resnet


class CosFace(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, 0.02, size=(in_size, out_size)))
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(x)
        x = F.normalize(x, dim=1)
        x = x @ F.normalize(self.W, dim=0)
        return x


class ResCosFace(nn.Module):
    def __init__(self, num_classes, emb_size=1024):
        super().__init__()
        self.resnet = resnet.ResNet18(emb_size)
        self.norm = CosFace(emb_size, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.norm(x)
        return x


class Res34CosFace(nn.Module):
    def __init__(self, num_classes, emb_size=1024):
        super().__init__()
        self.resnet = resnet.ResNet34(emb_size)
        self.norm = CosFace(emb_size, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = self.norm(x)
        return x
