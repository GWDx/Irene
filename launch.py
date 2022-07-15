from net import *
from go import *
import torch
import matplotlib.pyplot as plt
from features import getAllFeatures

# load net.pt
net = Net()
net.load_state_dict(torch.load('net.pt'))

go = Go()

count = 1

while True:
    if count % 2 == 1:
        turn = 1
    else:
        turn = -1

    features = getAllFeatures(go, turn)
    features = torch.tensor(features).bool()
    predict = net(features)
    predictIndex = torch.argmax(predict)
    x, y = toPosition(predictIndex)

    if go.move(turn, x, y) == False:
        raise Exception('Invalid move')

    # export image
    plt.imshow(go.board)
    plt.savefig(f'image/{count}.png')

    count += 1
