import math
import torch
import torch.nn as nn
import torchvision.models as models

import sys
sys.path.append("..")

from utils.functional import l2_norm

class ResNet(nn.Module):
    def __init__(self, n_class=3, use_norm=True):
        super().__init__()
        self.n_class = int(n_class)
        self.use_norm = use_norm
        resnet = models.resnet50(pretrained=True)


        self.model = nn.Sequential(
                            resnet.conv1,
                            resnet.bn1,
                            resnet.relu,
                            resnet.maxpool,

                            resnet.layer1,
                            resnet.layer2,
                            resnet.layer3,
                            resnet.layer4,

                            resnet.avgpool,
                            )
        
        self.dropout = nn.Dropout(0.2)
        if self.use_norm:
            self.w = nn.Parameter(torch.Tensor(2048, self.n_class))
        else:
            self.fc_angles = nn.Linear(2048, self.n_class)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)

        if self.use_norm:
            return torch.matmul(l2_norm(x, 1), l2_norm(self.w, 0)) * 180

        x = self.dropout(x)
        
        angles = self.fc_angles(x)
        
        return angles
