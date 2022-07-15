from sgfmill import sgf
from go import *
from prepareData import *
from net import *

# use cuda if available
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def loadData():
    inputData, outputData = torch.load('data.pt')
    return inputData, outputData


net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_function = nn.NLLLoss()


def train(epoch=10):
    inputData, outputData = loadData()

    # randomize data
    length = len(inputData)
    trainLength = int(length * 0.8)
    trainInputData, testInputData = inputData[:trainLength], inputData[trainLength:]
    trainOutputData, testOutputData = outputData[:trainLength], outputData[trainLength:]

    # trainInputData, trainOutputData = inputData, outputData

    # use cuda to train
    net.to(device)

    # batch size = 10
    batchSize = 100
    batchCount = int(len(trainInputData) / batchSize)

    logNumber = 100

    testBatchSize = 1000
    testBatchCount = int(len(testInputData) / testBatchSize)
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
            correctCount = torch.sum(torch.argmax(output, dim=1) == outputDataBatch).item()
            totalCorrectCount += correctCount

            # backward
            loss = loss_function(output, outputDataBatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

            # print(loss.item())
            # print(correctCount)
            # print
            if i % logNumber == 0 and i != 0:
                correctRate = totalCorrectCount / (logNumber * batchSize)
                avgLoss = totalLoss / logNumber
                # print('epoch:', epoch, 'batch:', i:5, 'correctRate:', correctRate, 'avgLoss:', avgLoss)
                print(f'epoch: {epoch:5} batch: {i:>5} correctRate: {correctRate:.2%} avgLoss: {avgLoss:.2f}')
                totalCorrectCount = 0
                totalLoss = 0

        # test
        with torch.no_grad():
            for i in range(testBatchCount):
                testInputDataBatch = testInputData[i * testBatchSize:(i + 1) * testBatchSize]
                testOutputDataBatch = testOutputData[i * testBatchSize:(i + 1) * testBatchSize].reshape(-1)

                testInputDataBatch = testInputDataBatch.to(device)
                testOutputDataBatch = testOutputDataBatch.to(device)

                output = net(testInputDataBatch)
                correctCount = torch.sum(torch.argmax(output, dim=1) == testOutputDataBatch).item()
                totalCorrectCount += correctCount

                loss = loss_function(output, testOutputDataBatch)
                totalLoss += loss.item()

            correctRate = totalCorrectCount / len(testInputData)
            avgLoss = totalLoss / testBatchCount
            print(f'epoch: {epoch:5}              correctRate: {correctRate:.2%} avgLoss: {avgLoss:.2f}')

        # save net
        torch.save(net.state_dict(), 'net.pt')


train(5)


def testFinal():
    inputData = torch.zeros(1, 1, 19, 19).int().to(device)

    output = net(inputData)

    outputIndex = torch.argmax(output)
    x, y = toPosition(outputIndex)

    print(x, y)


testFinal()
