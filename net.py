import torch
import torch.nn as nn
import torch.nn.functional as F
from go import *


# 1x19x19 [0,1,2]
# 48x19x19
# Conv 1x19x19 -> 48x19x19
# output: 1x19x19 double
# softmax
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 96, kernel_size=3, stride=1, padding=1)

        # deconvolution
        self.conv3 = nn.ConvTranspose2d(96, 24, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose2d(24, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        blank = x == 0
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = blank.float() * x
        x = x.view(-1, 19 * 19)
        x = F.log_softmax(x, dim=1)
        return x
