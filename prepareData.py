from sgfmill import sgf
from go import *
import torch
import matplotlib.pyplot as plt
from numba import jit
from features import getAllFeatures

colorCharToIndex = {'B': 1, 'W': -1, 'b': 1, 'w': -1}


# @jit
def prepareSgfFile(fileName):
    with open(fileName, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    sequence = game.get_main_sequence()

    winnerChar = game.get_winner()
    winner = colorCharToIndex[winnerChar]

    validSequence = []
    for node in sequence:
        # print(node.get_move())
        move = node.get_move()
        if move[1]:
            validSequence.append(move)

    go = Go()

    # append go.board to inputData
    inputData = []
    policyOutput = []
    valueOutput = []

    for move in validSequence:
        willPlayColor = colorCharToIndex[move[0]]
        x = move[1][0]
        y = move[1][1]
        inputData.append(getAllFeatures(go, willPlayColor))
        policyOutput.append(toDigit(x, y))
        valueOutput.append(winner == willPlayColor)

        if go.move(willPlayColor, x, y) == False:
            raise Exception('Invalid move')

    # use torch to load data
    inputData = torch.tensor(np.array(inputData)).bool()
    policyOutput = torch.tensor(np.array(policyOutput)).long().reshape(-1)
    valueOutput = torch.tensor(np.array(valueOutput)).long().reshape(-1)

    return inputData, policyOutput, valueOutput


def prepareData():
    # open jgdb/allValid.txt to read each line as sgfFile
    with open('jgdb/allValid.txt', 'r') as allValidFile:
        allValidLines = allValidFile.readlines()

    allInputData = []
    allPolicyOutput = []
    allValueOutput = []
    fileCount = 2000

    for sgfFile in allValidLines[:fileCount]:
        try:
            sgfFile = sgfFile.strip()
            inputData, policyOutput, valueOutput = prepareSgfFile(sgfFile)
            allInputData.append(inputData)
            allPolicyOutput.append(policyOutput)
            allValueOutput.append(valueOutput)

        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            print('Error: ' + sgfFile)

    allInputData = torch.cat(allInputData)
    allPolicyOutput = torch.cat(allPolicyOutput)
    allValueOutput = torch.cat(allValueOutput)

    # allInputData, allpolicyOutput = prepareSgfFile('test.sgf')
    # allInputData = allInputData.reshape(-1, 1, 19, 19)

    # plt.imshow(allInputData[1][0])
    # print(toPosition(allpolicyOutput[0]))
    # print(toPosition(allpolicyOutput[1]))
    # plt.show()

    torch.save((allInputData, allPolicyOutput), 'policyData.pt')
    torch.save((allInputData, allValueOutput), 'valueData.pt')


if __name__ == '__main__':
    prepareData()
