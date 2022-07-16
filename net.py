from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from go import *


# 19 19 28
# conv
# relu
# 19 19 32
# conv
# relu
# 19 19 32
# conv
# relu
# 19 19 32
# conv
# relu
# 19 19 32
# conv
# 19 19 1
# +b
# softmax
# Cross entropy
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 1, 3, padding=1)
        self.linear = nn.Linear(19 * 19, 19 * 19)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        x = x.view(-1, 19 * 19)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
