from net import *
from go import *
from features import getAllFeatures
import torch
import sys

# load net.pt
policyNet = PolicyNetwork()
policyNet.load_state_dict(torch.load('/home/gwd/文档/Resourses/Irene/policyNet.pt'))

valueNet = ValueNetwork()
valueNet.load_state_dict(torch.load('/home/gwd/文档/Resourses/Irene/valueNet.pt'))

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

colorCharToIndex = {'B': 1, 'W': -1, 'b': 1, 'w': -1}

while True:
    # implement GTP (Go Text Protocol)
    line = input().strip()
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

            color = colorCharToIndex[color]

            if go.move(-1, x, y) == False:
                print('Illegal move')
            else:
                print('ok')
    elif line.startswith('genmove'):
        colorChar = line.split()[1]
        willPlayColor = colorCharToIndex[colorChar]
        features = getAllFeatures(go, willPlayColor)
        features = torch.tensor(features).bool().reshape(1, -1, 19, 19)
        predict = policyNet(features)[0]
        predictReverseSortIndex = reversed(torch.argsort(predict))

        # sys err valueNet output
        value = valueNet(features)[0].item()
        sys.stderr.write(f'{colorChar} {value}\n')

        with open('/home/gwd/文档/Resourses/Irene/valueOutput.txt', 'a') as f:
            f.write(f'{colorChar} {value}\n')

        for predictIndex in predictReverseSortIndex:
            x, y = toPosition(predictIndex)
            moveResult = go.move(willPlayColor, x, y)

            x = 19 - x
            y = indexToChar[y]

            if moveResult == False:
                sys.stderr.write(f'Illegal move: {y}{x}\n')
            else:
                print(f'{y}{x}')
                break

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
