import torch
from torch import nn
from torchvision import models
from .efficientnet import EfficientNet

class COVIDNext50(nn.Module):
    def __init__(self, n_classes):
        super(COVIDNext50, self).__init__()
        self.n_classes = n_classes
        trainable = True
        self.efficientnet = EfficientNet(7, pretrained=True)

    def forward(self, input):
        return self.efficientnet(input)

    def probability(self, logits):
        return nn.functional.softmax(logits, dim=-1)
