"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from base.resnetv1b import resnet50_v1b, resnet101_v1b, resnet152_v1b, resnet34_v1b

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()





class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, backbone='resnet34', pretrained_base=False, dilated=False, **kwargs):
        super(SegBaseModel, self).__init__()
        if backbone == "resnet34":
            self.backbone = resnet34_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.conv1_channel = 64
            self.base_channel = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = resnet50_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.conv1_channel = 64
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.backbone = resnet101_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.conv1_channel = 64
            self.base_channel = [256, 512, 1024, 2048]
        elif backbone == 'resnet152':
            self.backbone = resnet152_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
            self.conv1_channel = 64
            self.base_channel = [256, 512, 1024, 2048]
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        if pretrained_base == False:
            initialize_weights(self)


    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)

        return c1, c2, c3, c4
