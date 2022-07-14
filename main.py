import os
import time
from sgfmill import sgf
from go import Go
import matplotlib.pyplot as plt

with open('test.sgf', 'rb') as f:
    game = sgf.Sgf_game.from_bytes(f.read())
sequence = game.get_main_sequence()

# (None, None)
# ('w', (2, 14))
# ('b', (16, 3))
# ('w', (14, 3))
# ('b', (14, 2))

# delete None moves
validSequence = []
for node in sequence:
    # print(node.get_move())
    move = node.get_move()
    if move[1]:
        validSequence.append(move)

for move in validSequence:
    print(move)

go = Go()

index = 0

for move in validSequence:
    if move[0] == 'w':
        color = 1
    else:
        color = 2
    x = move[1][0]
    y = move[1][1]
    go.move(color, x, y)
    print(go.board)
    # # use matplotlib to visualize the board
    # plt.imshow(go.board)
    # # save the image
    # plt.savefig('image/{}.png'.format(index))
    # index += 1
