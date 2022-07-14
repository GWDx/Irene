import os
import time
from sgfmill import sgf
from go import Go

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

start = time.time()

for move in validSequence:
    if move[0] == 'w':
        color = 1
    else:
        color = 2
    x = move[1][0]
    y = move[1][1]
    go.move(color, x, y)
    # print(go.board)

end = time.time()
print('time:', end - start)
