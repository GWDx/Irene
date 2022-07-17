from net import *
from go import *
import sys
from features import getAllFeatures

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
    predict = policyNet(features)[0]
    predictReverseSortIndex = reversed(torch.argsort(predict))

    # sys err valueNet output
    value = valueNet(features)[0].item()
    sys.stderr.write(f'{colorChar} {value}\n')

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
    def __init__(self, go, color, parent=None):
        self.go = go
        self.color = color
        self.parent = parent
        self.children = []
        self.N = 0  # visit count
        self.Q = 0  # win rate
        self.P = 0  # prior probability
        self.expanded = False

    def UCB(self):
        if self.N == 0:
            return float('inf')
        return self.Q / self.N + np.sqrt(np.log(self.parent.N) / self.N)


def getPlayoutResult(go, color):
    inputData = getAllFeatures(go, color)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    predict = playoutNet(inputData)[0]
    return predict


def getValueResult(go, color):
    inputData = getAllFeatures(go, color)
    inputData = torch.tensor(inputData).bool().reshape(1, -1, 19, 19)
    predict = valueNet(inputData)[0].item()
    return predict


# 选取 UCB 最大的节点
def bestChild(node):
    bestChild = None
    bestUCB = float('-inf')
    for child in node.children:
        ucb = child.UCB()
        if ucb > bestUCB:
            bestChild = child
            bestUCB = ucb
    return bestChild


# 扩展一次
def expandOne(go, node, color):
    predict = defaultPolicy(go, node, color)

    while True:
        moveIndex = np.random.choice(19 * 19, p=predict.numpy())
        x, y = toPosition(moveIndex)
        newGo = go.clone()

        if newGo.move(color, x, y):
            return MCTSNode(newGo, -color, node)


# 随机操作后创建新的节点，返回最终节点的 value
def defaultPolicy(go, parent, color):
    newGo = go.clone()
    currentColor = color
    for i in range(10):
        predict = getPlayoutResult(newGo, currentColor)
        x, y = toPosition(torch.argmax(predict))
        newGo.move(currentColor, x, y)  # TODO
        currentColor = -currentColor

    return predict


# 传入当前开始搜索的节点，返回创建的新的节点
# 先找当前未选择过的子节点，如果有多个则随机选。如果都选择过就找 UCB 最大的节点
def treePolicy(go, color):
    node = MCTSNode(go, color)

    predict = getPlayoutResult(go, color)
    predictReverseSortIndex = reversed(torch.argsort(predict))
    count = 0

    for predictIndex in predictReverseSortIndex:
        x, y = toPosition(predictIndex)
        newGo = go.clone()

        if newGo.move(color, x, y):
            newNode = MCTSNode(newGo, -color, node)
            node.children.append(newNode)
            count += 1
            if count == 5:
                break
    node.expanded = True

    allExpanded = True
    for child in node.children:
        if not child.expanded:
            allExpanded = False
            break

    if allExpanded:
        return bestChild(node)
    else:
        return node


def backward(node, value):
    while node:
        node.N += 1
        node.Q += value
        node = node.parent


def MCTS(node):
    for i in range(100):
        expandNode = treePolicy(node.go, node.color)
        value = defaultPolicy(expandNode.go, expandNode, expandNode.color)
        backward(expandNode, value)

    bestNextNode = bestChild(node)
    return bestNextNode


def genMoveMCTS(go, willPlayColor):
    node = MCTSNode(go, willPlayColor)
    bestNextNode = MCTS(node)
    bestMove = bestNextNode.go.history[-1]

    x, y = toPosition(bestMove)
    moveResult = bestNextNode.go.move(willPlayColor, x, y)

    x = 19 - x
    y = indexToChar[y]

    if moveResult == False:
        sys.stderr.write(f'Illegal move: {y}{x}\n')
    else:
        print(f'{y}{x}')


if __name__ == '__main__':
    # 初始化棋盘
    go = Go()

    # MCTS
    willPlayColor = 1
    genMoveMCTS(go, willPlayColor)
