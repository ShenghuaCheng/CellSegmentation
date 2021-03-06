import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MILResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MILResNet, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # encoder ????????????
        self.avgpool_patch = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_patch = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool_slide = nn.AdaptiveAvgPool2d((5, 5))
        self.fc_slide_cls = nn.Linear(512 * 5 * 5 * block.expansion, 2)
        # ?????????????????? AlexNet ?????????
        self.fc_slide_reg = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 5 * 5 * block.expansion, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        self.image_channels = 32  # slide mode ????????????????????????????????????

        # ???????????????
        self.pyramid_10 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.image_channels, kernel_size=1, stride=1),
        )
        self.pyramid_19 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.image_channels, kernel_size=1, stride=1),
        )
        self.pyramid_38 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.image_channels, kernel_size=1, stride=1),
        )

        # ????????????????????????
        self.upsample_conv1 = nn.Sequential(
            nn.Conv2d(2 * self.image_channels, self.image_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.image_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample_conv2 = nn.Sequential(
            nn.Conv2d(2 * self.image_channels, self.image_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.image_channels),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.mode == "patch":

            x = self.avgpool_patch(x4)  # x: [N * k, 512, 1, 1]
            x = self.fc_patch(torch.flatten(x, 1))  # x: [N * k, 512]

            return x

        elif self.mode == "slide":

            # slide_cls & slide_reg
            out = self.avgpool_slide(x4)  # [N, 512, 5, 5]
            out_cls = self.fc_slide_cls(torch.flatten(out, 1))  # [N, 2]
            out_reg = self.fc_slide_reg(torch.flatten(out, 1))  # [N, 1]

            # slide_seg
            out_x4 = self.pyramid_10(x4)  # out_x4: [N, 32, 10, 10]
            out_x3 = self.pyramid_19(x3)  # out_x3: [N, 32, 19, 19]
            out_x2 = self.pyramid_38(x2)  # out_x2: [N, 32, 38, 38]
            out_seg = F.interpolate(out_x4.clone(), size=19, mode="bilinear", align_corners=True)
            out_seg = torch.cat([out_seg, out_x3], dim=1)  # ????????????
            out_seg = self.upsample_conv1(out_seg)  # ?????? x4 ??? x3 ?????????
            out_seg = F.interpolate(out_seg.clone(), size=38, mode="bilinear", align_corners=True)
            out_seg = torch.cat([out_seg, out_x2], dim=1)
            out_seg = self.upsample_conv2(out_seg)  # [N, 32, 38, 38]

            return out_cls, out_reg, out_seg

        else:
            raise Exception("Something wrong in setmode.")

    def setmode(self, mode):
        self.mode = mode

def MILresnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MILResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MILResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model
