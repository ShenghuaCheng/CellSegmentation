import torch
import torch.nn as nn
from torch.utils import model_zoo
from .resnet import BasicBlock, Bottleneck, ResNet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Encoder(ResNet):

    def __init__(self, block, layers):
        super(Encoder, self).__init__(block, layers)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # 共享编码器仅包含所有的卷积层，不包含全局池化和全连接层
        # x = self.avgpool(x4)
        # x = torch.flatten(x,1)
        # x = self.fc(x)
        # return x
        return x4

def encoder18(pretrained=False, **kwargs):
    encoder = Encoder(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        encoder.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return encoder

