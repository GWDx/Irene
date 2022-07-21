from go import *
from prepareData import *
from net import *
import sys

# use cuda if available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# set random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


def splitData(inputData, outputData, ratio):
    length = len(inputData)
    trainLength = int(length * ratio)
    trainInputData, testInputData = inputData[:trainLength], inputData[trainLength:]
    trainOutputData, testOutputData = outputData[:trainLength], outputData[trainLength:]

    trainPermutation = torch.randperm(len(trainInputData))
    trainInputData = trainInputData[trainPermutation]
    trainOutputData = trainOutputData[trainPermutation]

    testPermutation = torch.randperm(len(testInputData))
    testInputData = testInputData[testPermutation]
    testOutputData = testOutputData[testPermutation]

    return trainInputData, trainOutputData, testInputData, testOutputData


def trainPolicy(net, outputFileName, epoch=10):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    loss_function = nn.NLLLoss()

    inputData, outputData = torch.load('policyData.pt')

    trainInputData, trainOutputData, testInputData, testOutputData = splitData(inputData, outputData, 0.8)

    # use cuda to train
    net.to(device)

    # batch size = 10
    batchSize = 100
    trainBatchCount = int(len(trainInputData) / batchSize)

    logInterval = 100

    testBatchCount = int(len(testInputData) / batchSize)
    totalLoss = 0
    totalCorrectCount = 0

    for epoch in range(epoch):
        totalLoss = 0
        totalCorrectCount = 0

        for i in range(trainBatchCount):
            # get batch data
            inputDataBatch = trainInputData[i * batchSize:(i + 1) * batchSize]
            outputDataBatch = trainOutputData[i * batchSize:(i + 1) * batchSize].reshape(-1)

            # use cuda to train
            inputDataBatch = inputDataBatch.to(device)
            outputDataBatch = outputDataBatch.to(device)

            # forward
            output = net(inputDataBatch)
            correctCount = torch.sum(torch.argmax(output, dim=1) == outputDataBatch).item()
            totalCorrectCount += correctCount

            # backward
            loss = loss_function(output, outputDataBatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

            if i % logInterval == 0 and i != 0:
                correctRate = totalCorrectCount / (logInterval * batchSize)
                avgLoss = totalLoss / logInterval
                print(f'epoch: {epoch:3}   batch: {i:>5}   correctRate: {correctRate:.2%}   avgLoss: {avgLoss:.2f}')
                totalCorrectCount = 0
                totalLoss = 0

        scheduler.step()

        totalCorrectCount = 0
        totalLoss = 0

        # test
        with torch.no_grad():
            for i in range(testBatchCount):
                testInputDataBatch = testInputData[i * batchSize:(i + 1) * batchSize]
                testOutputDataBatch = testOutputData[i * batchSize:(i + 1) * batchSize].reshape(-1)

                testInputDataBatch = testInputDataBatch.to(device)
                testOutputDataBatch = testOutputDataBatch.to(device)

                output = net(testInputDataBatch)
                correctCount = torch.sum(torch.argmax(output, dim=1) == testOutputDataBatch).item()
                totalCorrectCount += correctCount

                loss = loss_function(output, testOutputDataBatch)
                totalLoss += loss.item()

            correctRate = totalCorrectCount / len(testInputData)
            avgLoss = totalLoss / len(testInputData) * batchSize
            learningRate = optimizer.param_groups[0]['lr']
            print(f'epoch: {epoch:3}                  correctRate: {correctRate:>2.2%}   avgLoss: {avgLoss:.2f}   '
                  f'learningRate: {learningRate}')
        # save net
        torch.save(net.state_dict(), outputFileName)


# valueData
def trainValue(net, outputFileName, epoch=10):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    loss_function = nn.MSELoss()

    inputData, outputData = torch.load('valueData.pt')

    # selectInterval = 5
    # inputData = inputData[::selectInterval]
    # outputData = outputData[::selectInterval]

    trainInputData, trainOutputData, testInputData, testOutputData = splitData(inputData, outputData, 0.8)

    # use cuda to train
    net.to(device)

    # batch size
    batchSize = 100
    batchCount = int(len(trainInputData) / batchSize)

    logInterval = 100

    testBatchCount = int(len(testInputData) / batchSize)
    totalLoss = 0
    totalCorrectCount = 0

    for epoch in range(epoch):
        totalLoss = 0
        totalCorrectCount = 0

        for i in range(batchCount):
            # get batch data
            inputDataBatch = trainInputData[i * batchSize:(i + 1) * batchSize]
            outputDataBatch = trainOutputData[i * batchSize:(i + 1) * batchSize].reshape(-1)

            # use cuda to train
            inputDataBatch = inputDataBatch.to(device)
            outputDataBatch = outputDataBatch.to(device)

            # forward
            output = net(inputDataBatch)
            outputInt = torch.round(output)
            correctCount = torch.sum(outputInt == outputDataBatch).item()
            totalCorrectCount += correctCount

            # backward
            loss = loss_function(output, outputDataBatch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

            # print
            if i % logInterval == 0 and i != 0:
                avgLoss = totalLoss / logInterval
                correctRate = totalCorrectCount / (logInterval * batchSize)
                print(f'epoch: {epoch:3}   batch: {i:>5}   correctRate: {correctRate:.2%}   avgLoss: {avgLoss:.2f}')
                totalCorrectCount = 0
                totalLoss = 0

        scheduler.step()

        totalCorrectCount = 0
        totalLoss = 0

        # test
        with torch.no_grad():
            for i in range(testBatchCount):
                testInputDataBatch = testInputData[i * batchSize:(i + 1) * batchSize]
                testOutputDataBatch = testOutputData[i * batchSize:(i + 1) * batchSize].reshape(-1)

                testInputDataBatch = testInputDataBatch.to(device)
                testOutputDataBatch = testOutputDataBatch.to(device)

                output = net(testInputDataBatch)
                outputInt = torch.round(output)
                correctCount = torch.sum(outputInt == testOutputDataBatch).item()
                totalCorrectCount += correctCount

                loss = loss_function(output, testOutputDataBatch)
                totalLoss += loss.item()

            correctRate = totalCorrectCount / len(testInputData)
            avgLoss = totalLoss / len(testInputData) * batchSize
            learningRate = optimizer.param_groups[0]['lr']
            print(f'epoch: {epoch:3}                  correctRate: {correctRate:>2.2%}   avgLoss: {avgLoss:.2f}   '
                  f'learningRate: {learningRate}')
        # save net
        torch.save(net.state_dict(), outputFileName)


# python3 train.py policyNet
if len(sys.argv) == 2:
    if sys.argv[1] == 'policyNet':
        net = PolicyNetwork()
        trainPolicy(net, 'policyNet.pt', 10)
    elif sys.argv[1] == 'playoutNet':
        net = PlayoutNetwork()
        trainPolicy(net, 'playoutNet.pt', 5)
    elif sys.argv[1] == 'valueNet':
        net = ValueNetwork()
        trainValue(net, 'valueNet.pt', 8)
else:
    net = ValueNetwork()
    trainValue(net, 'valueNet.pt', 8)
