import os
from sgfmill import sgf
from go import *
import torch
import matplotlib.pyplot as plt


def prepareSgfFile(fileName):
    with open(fileName, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    sequence = game.get_main_sequence()

    validSequence = []
    for node in sequence:
        # print(node.get_move())
        move = node.get_move()
        if move[1]:
            validSequence.append(move)

    go = Go()

    # append go.board to inputData
    inputData = []
    outputData = []

    for move in validSequence:
        if move[0] == 'b':
            color = 1
        else:
            color = 2
        x = move[1][0]
        y = move[1][1]
        board = np.array(go.board)
        if color == 2:
            for i in range(19):
                for j in range(19):
                    if board[i, j] > 0:
                        board[i, j] = 3 - board[i, j]
        inputData.append(board)
        outputData.append(toDigit(x, y))

        if go.move(color, x, y) == False:
            raise Exception('Invalid move')

    # use torch to load data
    inputData = torch.tensor(np.array(inputData)).reshape(-1, 19, 19)
    outputData = torch.tensor(np.array(outputData)).long().reshape(-1)

    return inputData, outputData


def prepareData():
    # open jgdb/allValid.txt to read each line as sgfFile
    with open('jgdb/allValid.txt', 'r') as allValidFile:
        allValidLines = allValidFile.readlines()

    allInputData = []
    allOutputData = []

    count = 0
    for sgfFile in allValidLines:
        try:
            sgfFile = sgfFile.strip()
            inputData, outputData = prepareSgfFile(sgfFile)
            allInputData.append(inputData)
            allOutputData.append(outputData)

            count += 1
            if count % 1000 == 0:
                break
                print(f'Processed {count} files')
        except:
            print('Error: ' + sgfFile)

    allInputData = torch.cat(allInputData).reshape(-1, 1, 19, 19)
    allOutputData = torch.cat(allOutputData)

    # allInputData, allOutputData = prepareSgfFile('test.sgf')
    # allInputData = allInputData.reshape(-1, 1, 19, 19)

    plt.imshow(allInputData[1].squeeze())
    print(toPosition(allOutputData[0]))
    print(toPosition(allOutputData[1]))
    # plt.show()

    torch.save((allInputData, allOutputData), 'data.pt')


if __name__ == '__main__':
    prepareData()
