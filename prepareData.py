from sgfmill import sgf
from go import *
import torch
import matplotlib.pyplot as plt
from numba import jit
from features import getAllFeatures


# @jit
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
            color = -1
        x = move[1][0]
        y = move[1][1]
        inputData.append(getAllFeatures(go, color))
        outputData.append(toDigit(x, y))

        if go.move(color, x, y) == False:
            raise Exception('Invalid move')

    # use torch to load data
    inputData = torch.tensor(np.array(inputData)).bool()
    outputData = torch.tensor(np.array(outputData)).long().reshape(-1)

    return inputData, outputData


def prepareData():
    # open jgdb/allValid.txt to read each line as sgfFile
    with open('jgdb/allValid.txt', 'r') as allValidFile:
        allValidLines = allValidFile.readlines()

    allInputData = []
    allOutputData = []
    fileCount = 2000

    for sgfFile in allValidLines[:fileCount]:
        try:
            sgfFile = sgfFile.strip()
            inputData, outputData = prepareSgfFile(sgfFile)
            allInputData.append(inputData)
            allOutputData.append(outputData)

        except KeyboardInterrupt:
            exit()
        except Exception:
            print('Error: ' + sgfFile)

    allInputData = torch.cat(allInputData)
    allOutputData = torch.cat(allOutputData)

    # allInputData, allOutputData = prepareSgfFile('test.sgf')
    # allInputData = allInputData.reshape(-1, 1, 19, 19)

    # plt.imshow(allInputData[1][0])
    # print(toPosition(allOutputData[0]))
    # print(toPosition(allOutputData[1]))
    # plt.show()

    torch.save((allInputData, allOutputData), 'data.pt')


if __name__ == '__main__':
    prepareData()
