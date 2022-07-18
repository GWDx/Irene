from unittest import result
from net import *
from go import *
import sys
from features import getAllFeatures

# set random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

# load net.pt
policyNet = PolicyNetwork()
policyNet.load_state_dict(torch.load('/home/gwd/文档/Resourses/Irene/policyNet.pt'))

playoutNet = PlayoutNetwork()
playoutNet.load_state_dict(torch.load('/home/gwd/文档/Resourses/Irene/playoutNet.pt'))

valueNet = ValueNetwork()
valueNet.load_state_dict(torch.load('/home/gwd/文档/Resourses/Irene/valueNet.pt'))

colorCharToIndex = {'B': 1, 'W': -1, 'b': 1, 'w': -1}
indexToColorChar = {1: 'B', -1: 'W'}
indexToChar = []
charToIndex = {}
char = ord('A')

for i in range(19):
    indexToChar.append(chr(char))
    charToIndex[chr(char)] = i
    char += 1
    if char == ord('I'):
        char += 1


def genMovePolicy(go, willPlayColor):
    colorChar = indexToColorChar[willPlayColor]

    features = getAllFeatures(go, willPlayColor)
    features = torch.tensor(features).bool().reshape(1, -1, 19, 19)
    predict = playoutNet(features)[0]
    predictReverseSortIndex = reversed(torch.argsort(predict))

    # sys err valueNet output
    # value = valueNet(features)[0].item()
    # sys.stderr.write(f'{colorChar} {value}\n')

    # with open('/home/gwd/文档/Resourses/Irene/valueOutput.txt', 'a') as f:
    #     f.write(f'{colorChar} {value}\n')

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


class MCTSNode:
    def __init__(self, go, willPlayColor, parent):
        self.go = go.clone()
        self.color = willPlayColor
        self.parent = parent
        self.children = []
        self.N = 0  # visit count
        self.Q = 0  # win rate
        self.P = 0  # prior probability
        self.expanded = False
        if parent:
            self.parent.children.append(self)

    def UCB(self):
        if self.N == 0:
            return float('-inf')
        return self.Q / self.N + np.sqrt(np.log(self.parent.N) / self.N)

    def __str__(self):
        lastPosition = self.go.history[-1]
        # result = f'{self.color} {self.N} {self.Q} {self.P}'
        # node = self
        # while node.parent:
        #     result += f' {self.parent.N}'
        #     node = node.parent
        result = f'{self.color} {self.N} {self.Q} {self.P} {self.UCB()} {lastPosition}'
        return result


def getPlayoutResult(go, willPlayColor):
    inputData = getAllFeatures(go, willPlayColor)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    predict = playoutNet(inputData)[0]
    return predict


def getValueResult(go, willPlayColor):
    inputData = getAllFeatures(go, willPlayColor)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    predict = valueNet(inputData)[0].item()
    return predict


# 选取 UCB 最大的节点
def getBestChild(node):
    # print([i.UCB() for i in node.children])
    # print([i.N for i in node.children])
    bestChild = None
    bestUCB = float('-inf')
    for child in node.children:
        ucb = child.UCB()
        if ucb > bestUCB:
            bestChild = child
            bestUCB = ucb
    # if debug:
    #     print(f'bestChild: {bestChild} bestUCB: {bestUCB}')
    return bestChild


def getMostVisitedChild(node):
    bestChild = None
    bestN = 0
    for child in node.children:
        if child.N > bestN:
            bestChild = child
            bestN = child.N
    return bestChild


# 随机操作后创建新的节点，返回最终节点的 value
def defaultPolicy(expandNode):
    newGo = expandNode.go.clone()
    willPlayColor = expandNode.color

    for i in range(10):
        predict = getPlayoutResult(newGo, willPlayColor)

        while True:
            # random choose a move
            selectedIndex = np.random.choice(len(predict), p=predict.exp().detach().numpy())
            x, y = toPosition(selectedIndex)
            if newGo.move(willPlayColor, x, y):
                break

        willPlayColor = -willPlayColor

    value = getValueResult(newGo, willPlayColor)

    if debug:
        print(f'expandNode: {expandNode} value: {value}')

    return value


def searchChildren(root):
    go = root.go
    nodeWillPlayColor = root.color

    predict = getPlayoutResult(go, nodeWillPlayColor)
    predictReverseSortIndex = reversed(torch.argsort(predict))

    count = 0
    nextColor = -nodeWillPlayColor

    for predictIndex in predictReverseSortIndex:
        x, y = toPosition(predictIndex)
        newGo = go.clone()

        if newGo.move(nodeWillPlayColor, x, y):
            newNode = MCTSNode(newGo, nextColor, root)
            count += 1
            if count == 3:
                break
    # node.expanded = True


# 传入当前开始搜索的节点，返回创建的新的节点
# 先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找 UCB 最大的节点
def treePolicy(root):
    allExpanded = True
    for child in root.children:
        if not child.expanded:
            allExpanded = False
            break

    if allExpanded:
        return getBestChild(root)
    else:
        return child


def backward(node, value):
    while node:
        node.N += 1
        node.Q += value
        node.expanded = True
        node = node.parent


def MCTS(root):
    for i in range(100):
        expandNode = treePolicy(root)
        value = defaultPolicy(expandNode)
        backward(expandNode, value)
        # print(expandNode)

    bestNextNode = getMostVisitedChild(root)
    return bestNextNode


def genMoveMCTS(go, willPlayColor):
    root = MCTSNode(go, willPlayColor, None)

    searchChildren(root)
    bestNextNode = MCTS(root)
    bestMove = bestNextNode.go.history[-1]

    if debug:
        playoutResult = getPlayoutResult(go, willPlayColor)
        playoutMove = toPosition(torch.argmax(playoutResult))
        print(playoutMove, bestMove, playoutMove == bestMove)
        for child in root.children:
            print(child)

    x, y = bestMove
    moveResult = go.move(willPlayColor, x, y)

    x = 19 - x
    y = indexToChar[y]

    if moveResult == False:
        sys.stderr.write(f'Illegal move: {y}{x}\n')
        exit(1)
    else:
        print(f'{y}{x}')
    return x, y


debug = False

if __name__ == '__main__':
    # 初始化棋盘
    go = Go()

    # willPlayColor = 1
    # for i in range(8):
    #     genMoveMCTS(go, willPlayColor)
    #     willPlayColor = -willPlayColor
    # debug = True
    # genMoveMCTS(go, willPlayColor)

    willPlayColor = 1

    go.move(1, 3, 16)
    go.move(-1, 3, 3)
    go.move(1, 16, 16)
    go.move(-1, 16, 3)

    debug = True
    genMoveMCTS(go, willPlayColor)
