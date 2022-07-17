import torch
from features import getAllFeatures
from go import *
from net import *

valueNet = ValueNetwork()
valueNet.load_state_dict(torch.load('valueNet.pt'))


def getValue(go, color):
    features = getAllFeatures(go, color)
    features = torch.tensor(features).bool().reshape(1, -1, 19, 19)
    predict = valueNet(features)[0]
    return predict.item()


go = Go()
nextColor = 1

print(getValue(go, nextColor))
