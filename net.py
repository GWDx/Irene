import torch
import torch.nn as nn
import torch.nn.functional as F
from go import *


# use CNN, 3 layers
# input: 1x19x19 [0,1,2]
# output: 1x19x19 double
# softmax
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 19 * 19, 19 * 19)

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        # x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 19 * 19)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
