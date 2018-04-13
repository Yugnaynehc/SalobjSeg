# -*- coding: utf-8 -*-

'''
Salient object segmentation by full conv net
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from args import resnet_checkpoint
from args import opt


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.resnet = models.resnet18()
        if opt.cuda:
            pmodel = torch.load(resnet_checkpoint)
        else:
            pmodel = torch.load(resnet_checkpoint, map_location=lambda storage, location: storage)
        self.resnet.load_state_dict(pmodel)
        del self.resnet.fc

        self.deconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.fc = nn.Conv2d(128, 1, 1)
        self.upsampling = nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self, images):
        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)  # 64 x 56 x 56
        x = self.resnet.layer2(x)  # 128 x 28 x 28
        x = self.resnet.layer3(x)  # 256 x 14 x 14
        x = self.resnet.layer4(x)  # 512 x 7 x 7
        x = self.deconv1(x, output_size=(14, 14))
        x = self.deconv2(x, output_size=(28, 28))
        x = self.fc(x)
        x = self.upsampling(x)
        return x
