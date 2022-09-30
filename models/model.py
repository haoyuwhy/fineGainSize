import torch.nn.functional as F
import torchvision
import torch
from torch import nn


def build_resnet(name="resnet50", pretrained=True):
    if name == "resnet34" or name == "resnet50" or name == "resnet101" or name == "resnet152":
        resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)
    else:
        resnet = torch.hub.load('XingangPan/IBN-Net', name, pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return resnet