from sgfmill import sgf
from go import *
from prepareData import *
from net import *

# use cuda if available
device = torch.device("cuda")


def loadData():
    inputData, outputData = torch.load('data.pt')
    return inputData, outputData


net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_function = nn.NLLLoss()


def train(epoch=10):
    inputData, outputData = loadData()

    # use cuda to train
    net.to(device)

    # batch size = 10
    batchSize = 100
    batchCount = int(len(inputData) / batchSize)

    correctCount = 0
    totalLoss = 0

    for epoch in range(epoch):
        for i in range(batchCount):
            # get batch data
            inputDataBatch = inputData[i * batchSize:(i + 1) * batchSize]
            outputDataBatch = outputData[i * batchSize:(i + 1) * batchSize].reshape(-1)

            # use cuda to train
            inputDataBatch = inputDataBatch.to(device)
            outputDataBatch = outputDataBatch.to(device)

            # forward
            output = net(inputDataBatch)
            correctCount += torch.sum(torch.argmax(output, dim=1) == outputDataBatch).item()

            # backward
            loss = loss_function(output, outputDataBatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()

            # print
            if i % 100 == 0:
                correctRate = correctCount / (100 * batchSize)
                avgLoss = totalLoss / 100
                # print('epoch:', epoch, 'batch:', i:5, 'correctRate:', correctRate, 'avgLoss:', avgLoss)
                print(f'epoch: {epoch:5} batch: {i:>5} correctRate: {correctRate:.2%} avgLoss: {avgLoss:.2f}')
                correctCount = 0
                totalLoss = 0

        # save net
        torch.save(net.state_dict(), 'net.pt')


train()


def test():
    inputData = torch.zeros(1, 1, 19, 19)

    output = net(inputData)

    outputIndex = torch.argmax(output)
    x, y = toPosition(outputIndex)

    print(x, y)


test()
