import numpy as np
import torch
import time
from sgfmill import sgf


class Go:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.liberty = np.zeros((size, size), dtype=np.int8)
        self.previousBoard = np.zeros((size, size), dtype=np.int8)
        self.history = [(None, None)] * 8

    def clone(self):
        go = Go(self.size)
        go.board = np.array(self.board)
        go.liberty = np.array(self.liberty)
        go.previousBoard = np.array(self.previousBoard)
        go.history = list(self.history)
        return go

    def move(self, color, x, y):
        # 0. 检查输入是否合法
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False

        # 1. 检查是否已经有棋子
        if self.board[x, y] != 0:
            return False

        anotherColor = -color
        # 2. 检查打劫
        if self.history[-2] == (x, y):
            # 如果周围全是对方的棋子
            if (x == 0 or self.board[x - 1, y] == anotherColor) and \
               (y == 0 or self.board[x, y - 1] == anotherColor) and \
               (x == self.size - 1 or self.board[x + 1, y] == anotherColor) and \
               (y == self.size - 1 or self.board[x, y + 1] == anotherColor):
                return False

        # 3. 落子，移除没有 liberty 的棋子
        self.previousBoard = np.array(self.board)

        self.board[x, y] = color

        if x > 0:
            self.clearColorNear(anotherColor, x - 1, y)
        if x < self.size - 1:
            self.clearColorNear(anotherColor, x + 1, y)
        if y > 0:
            self.clearColorNear(anotherColor, x, y - 1)
        if y < self.size - 1:
            self.clearColorNear(anotherColor, x, y + 1)

        self.clearColorNear(color, x, y)

        if self.board[x, y] == 0:
            self.board = self.previousBoard
            return False

        self.history.append((x, y))

        return True

    def clearColorNear(self, color, x, y):
        if self.board[x, y] != color:
            return

        visited = np.zeros((self.size, self.size), dtype=np.int32)
        boardGroup = np.zeros((self.size, self.size), dtype=np.int32)

        def dfs(colorBoard, x, y):
            if visited[x, y] == 1:
                return
            if colorBoard[x, y] == 0:
                if self.board[x, y] == 0:
                    allLibertyPosition.add((x, y))
                return
            visited[x, y] = 1
            boardGroup[x, y] = 1

            if x > 0:
                dfs(colorBoard, x - 1, y)
            if x < self.size - 1:
                dfs(colorBoard, x + 1, y)
            if y > 0:
                dfs(colorBoard, x, y - 1)
            if y < self.size - 1:
                dfs(colorBoard, x, y + 1)

        colorBoard = self.board == color

        allLibertyPosition = set()
        dfs(colorBoard, x, y)

        liberties = len(allLibertyPosition)
        # dead group
        if liberties == 0:
            self.board[boardGroup == 1] = 0
        else:
            self.liberty[boardGroup == 1] = liberties


def toDigit(x, y):
    return x * 19 + y


def toPosition(digit):
    if isinstance(digit, torch.Tensor):
        digit = digit.item()
    if digit == 361:
        return None, None
    x = digit // 19
    y = digit % 19
    return x, y


def testKill():
    go = Go()
    #     3 4 5 6
    # 15    B W
    # 16  B W B W
    # 17    B W
    go.move(1, 15, 4)
    assert go.move(2, 15, 4) == False

    go.move(-1, 15, 5)
    go.move(1, 16, 3)
    go.move(-1, 16, 4)
    go.move(1, 17, 4)
    go.move(-1, 17, 5)
    go.move(1, 16, 5)
    go.move(-1, 16, 6)

    print(go.board)

    assert go.board[16, 4] == 0

    go.move(1, 4, 4)
    go.move(-1, 16, 4)

    assert go.move(1, 16, 4) == False

    print(go.board)


def testLiberty():
    go = Go()
    go.move(1, 4, 4)
    go.move(1, 4, 5)
    go.move(1, 4, 6)
    go.move(1, 5, 4)
    go.move(-1, 5, 5)

    print(go.board)
    print(go.liberty)

    assert go.liberty[4, 4] == go.liberty[4, 5] == go.liberty[4, 6] == go.liberty[5, 4] == 8
    assert go.liberty[5, 5] == 2


def testTime():
    with open('test.sgf', 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    sequence = game.get_main_sequence()

    validSequence = []
    for node in sequence:
        # print(node.get_move())
        move = node.get_move()
        if move[1]:
            validSequence.append(move)
    # for move in validSequence:
    #     print(move)
    go = Go()

    start = time.time()
    for move in validSequence:
        if move[0] == 'w':
            color = 1
        else:
            color = 2
        x = move[1][0]
        y = move[1][1]
        go.move(color, x, y)
        # print(go.board)
    end = time.time()
    print('time:', end - start)


if __name__ == '__main__':
    testKill()
    testLiberty()
    testTime()
