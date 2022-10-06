from net import *
from go import *
import sys
import os
from features import getAllFeatures

# set random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

programPath = os.path.dirname(os.path.realpath(__file__))

# load net.pt
policyNet = PolicyNetwork()
policyNet.load_state_dict(torch.load(programPath + '/policyNet.pt'))

playoutNet = PlayoutNetwork()
playoutNet.load_state_dict(torch.load(programPath + '/playoutNet.pt'))

valueNet = ValueNetwork()
valueNet.load_state_dict(torch.load(programPath + '/valueNet.pt'))

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


def toStrPosition(x, y):
    if (x, y) == (None, None):
        # return 'pass'
        return ''
    x = 19 - x
    y = indexToChar[y]
    return f'{y}{x}'


def getPolicyNetResult(go, willPlayColor):
    inputData = getAllFeatures(go, willPlayColor)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    predict = policyNet(inputData)[0]
    return predict


def getPlayoutNetResult(go, willPlayColor):
    inputData = getAllFeatures(go, willPlayColor)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    predict = playoutNet(inputData)[0]
    return predict


def getValueNetResult(go, willPlayColor):
    inputData = getAllFeatures(go, willPlayColor)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    value = valueNet(inputData)[0].item()
    return value


def getValueResult(go, willPlayColor):
    # predict = playoutNet(inputData)[0, 361]
    # return 1 - predict.item()

    countThisColor = np.sum(go.board == willPlayColor)
    countAnotherColor = np.sum(go.board == -willPlayColor)
    return countThisColor - countAnotherColor


def genMovePolicy(go, willPlayColor):
    predict = getPolicyNetResult(go, willPlayColor)
    predictReverseSortIndex = reversed(torch.argsort(predict))

    # sys err valueNet output
    value = getValueResult(go, willPlayColor)
    sys.stderr.write(f'{willPlayColor} {value}\n')

    # with open('valueOutput.txt', 'a') as f:
    #     f.write(f'{colorChar} {value}\n')

    for predictIndex in predictReverseSortIndex:
        x, y = toPosition(predictIndex)
        if (x, y) == (None, None):
            print('pass')
            return
        moveResult = go.move(willPlayColor, x, y)
        strPosition = toStrPosition(x, y)

        if moveResult == False:
            sys.stderr.write(f'Illegal move: {strPosition}\n')
        else:
            print(strPosition)
            break


class MCTSNode:
    def __init__(self, go, willPlayColor, parent):
        self.go = go.clone()
        self.color = willPlayColor
        self.parent = parent
        self.children = []
        self.N = 0  # visit count
        self.Q = 0  # win rate
        self.expanded = False
        if parent:
            self.parent.children.append(self)

    def UCB(self):
        if self.N == 0:
            return float('-inf')
        return self.Q / self.N + np.sqrt(np.log(self.parent.N) / self.N)

    def __str__(self):
        x, y = self.go.history[-1]
        strPosition = toStrPosition(x, y)
        result = f'{self.color} {self.N} {self.Q} {self.UCB()} {strPosition}'
        return result


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
def defaultPolicy(expandNode, rootColor):
    newGo = expandNode.go.clone()
    willPlayColor = expandNode.color

    for i in range(5):
        predict = getPlayoutNetResult(newGo, willPlayColor)

        while True:
            # random choose a move
            selectedIndex = np.random.choice(len(predict), p=predict.exp().detach().numpy())
            x, y = toPosition(selectedIndex)
            if (x, y) == (None, None):
                continue
            if newGo.move(willPlayColor, x, y):
                break

        willPlayColor = -willPlayColor

    value = getValueNetResult(newGo, rootColor)

    if debug:
        print(f'expandNode: {expandNode} value: {value}')

    return value


def searchChildren(node):
    go = node.go
    nodeWillPlayColor = node.color

    predict = getPolicyNetResult(go, nodeWillPlayColor)
    predictReverseSortIndex = reversed(torch.argsort(predict))

    count = 0
    nextColor = -nodeWillPlayColor

    if predict[361].exp().item() > 0.5:
        print('pass')
        return

    for predictIndex in predictReverseSortIndex:
        x, y = toPosition(predictIndex)
        if (x, y) == (None, None):
            continue
        newGo = go.clone()

        if newGo.move(nodeWillPlayColor, x, y):
            newNode = MCTSNode(newGo, nextColor, node)
            count += 1
            if count == 2:
                break
    # node.expanded = True


# 传入当前开始搜索的节点，返回创建的新的节点
# 先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找 UCB 最大的节点
def treePolicy(root):
    node = root
    while True:
        if len(node.children) == 0:
            return node

        allExpanded = True
        for child in node.children:
            if not child.expanded:
                allExpanded = False
                break

        if allExpanded:
            node = getBestChild(node)
        else:
            return child


def backward(node, value):
    while node:
        node.N += 1
        node.Q += value
        node.expanded = True
        node = node.parent


def MCTS(root):
    rootColor = root.color
    for i in range(200):
        expandNode = treePolicy(root)
        assert expandNode != None
        searchChildren(expandNode)
        value = defaultPolicy(expandNode, rootColor)
        backward(expandNode, value)
        # print(expandNode)

    bestNextNode = getBestChild(root)
    return bestNextNode


def genMoveMCTS(go, willPlayColor):
    root = MCTSNode(go, willPlayColor, None)

    bestNextNode = MCTS(root)
    bestMove = bestNextNode.go.history[-1]

    if debug:
        playoutResult = getPlayoutNetResult(go, willPlayColor)
        playoutMove = toPosition(torch.argmax(playoutResult))
        print(playoutMove, bestMove, playoutMove == bestMove)
        for child in root.children:
            print(child)

    for child in root.children:
        sys.stderr.write(str(child) + '\n')

    x, y = bestMove
    moveResult = go.move(willPlayColor, x, y)
    strPosition = toStrPosition(x, y)

    if moveResult == False:
        sys.stderr.write(f'Illegal move: {strPosition}')
        exit(1)
    else:
        print(strPosition)
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

    go.move(1, 3, 16)
    go.move(-1, 3, 3)
    go.move(1, 16, 16)
    go.move(-1, 16, 3)
    go.move(1, 2, 5)

    debug = True
    genMoveMCTS(go, -1)

    for item in go.history:
        print(toStrPosition(item[0], item[1]))
