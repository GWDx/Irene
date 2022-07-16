from net import *
from go import *
from features import getAllFeatures
import torch
import sys

# load net.pt
net = Net()
net.load_state_dict(torch.load('/home/gwd/文档/Resourses/Irene/net.pt'))

go = Go()

count = 1

# stderr output 'GTP ready'
sys.stderr.write('GTP ready\n')

indexToChar = []
charToIndex = {}
char = ord('A')

for i in range(19):
    indexToChar.append(chr(char))
    charToIndex[chr(char)] = i
    char += 1
    if char == ord('I'):
        char += 1

while True:
    # implement GTP (Go Text Protocol)
    line = input()
    if line == 'quit':
        break
    print('= ', end='')
    if line == 'boardsize 19':
        print('boardsize 19')
    elif line.startswith('komi'):
        print('komi')
    if line == 'clear_board':
        go = Go()
        print('clear_board')
    elif line.startswith('play'):
        # play B F12
        color, position = line.split()[1:]
        if position == 'pass':
            print('play PASS')
        else:
            # position = F12
            y, x = position[0], position[1:]

            #    A B C D E F G H J K L M N O P Q R S T
            # 19
            # 18
            # 17

            x = 19 - int(x)
            y = charToIndex[y]

            color = 1 if color == 'B' else -1

            if go.move(-1, x, y) == False:
                print('Illegal move')
            else:
                print('ok')
    elif line.startswith('genmove'):
        turn = -1 if line.split()[1] == 'b' else 1
        features = getAllFeatures(go, turn)
        features = torch.tensor(features).bool().reshape(1, -1, 19, 19)
        predict = net(features)
        predictIndex = torch.argmax(predict)
        x, y = toPosition(predictIndex)

        if go.move(turn, x, y) == False:
            print('Wrong move')
        else:
            x = 19 - x
            y = indexToChar[y]
            print(f'{y}{x}')

    elif line.startswith('showboard'):
        for i in range(19):
            for j in range(19):
                if go.board[i][j] == 1:
                    print('X', end='')
                elif go.board[i][j] == -1:
                    print('O', end='')
                else:
                    print('.', end='')
            print()
    # name
    elif line.startswith('name'):
        print('Irene')
    # version
    elif line.startswith('version'):
        print('0.1')
    # protocol_version
    elif line.startswith('protocol_version'):
        print('2')
    # list_commands
    elif line.startswith('list_commands'):
        print('name')
        print('version')
        print('protocol_version')
        print('list_commands')
        print('clear_board')
        print('boardsize')
        print('showboard')
        print('play')
        print('genmove')
        print('quit')
    else:
        print('Unknown command')

    print()
