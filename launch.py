import gtp
from net import *
from go import *
import torch
import matplotlib.pyplot as plt

# load net.pt
net = Net()
net.load_state_dict(torch.load('net.pt'))

go = Go()

count = 1

while True:
    board = np.array(go.board)
    if count % 2 == 1:
        turn = 1
    else:
        turn = -1

    if turn == -1:
        board = -board
    inputData = torch.tensor(board).int().reshape(-1, 19, 19)
    predict = net(inputData)
    predictIndex = torch.argmax(predict)
    x, y = toPosition(predictIndex)

    if go.move(turn, x, y) == False:
        raise Exception('Invalid move')

    # export image
    plt.imshow(go.board)
    plt.savefig(f'image/{count}.png')

    count += 1
