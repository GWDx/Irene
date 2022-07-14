import numpy as np
import torch


class Go:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int32)
        self.previousMove = (-1, -1)  # 避免打劫
        self.previousBoard = np.zeros((size, size), dtype=np.int32)

    def move(self, color, x, y):
        # 检查是否合法
        # 1. 检查是否已经有棋子
        if self.board[x, y] > 0:
            return False

        # 2. 检查打劫
        if self.previousMove == (x, y):
            return False

        # 3. 落子，移除没有 liberty 的棋子
        self.previousMove = (x, y)
        self.previousBoard = self.board

        anotherColor = 3 - color
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
                    allLiberityPosition.add((x, y))
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

        allLiberityPosition = set()
        dfs(colorBoard, x, y)

        # dead group
        if len(allLiberityPosition) == 0:
            self.board[boardGroup == 1] = 0


def toDigit(x, y):
    return x * 19 + y


def toPosition(digit):
    if isinstance(digit, torch.Tensor):
        digit = digit.item()
    x = digit // 19
    y = digit % 19
    return x, y


if __name__ == '__main__':
    go = Go()
    #     3 4 5 6
    # 15    B W
    # 16  B W B W
    # 17    B W
    go.move(1, 15, 4)
    assert go.move(2, 15, 4) == False

    go.move(2, 15, 5)
    go.move(1, 16, 3)
    go.move(2, 16, 4)
    go.move(1, 17, 4)
    go.move(2, 17, 5)
    go.move(1, 16, 5)
    go.move(2, 16, 6)

    assert go.board[16, 4] == 0

    go.move(1, 4, 4)
    go.move(2, 16, 4)

    assert go.move(1, 16, 4) == False

    print(go.board)
