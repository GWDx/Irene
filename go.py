import numpy as np


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

        self.clearColor(anotherColor)
        self.clearColor(color)

        if self.board[x, y] == 0:
            self.board = self.previousBoard
            return False

        return True

    def clearColor(self, color):
        visited = np.zeros((self.size, self.size), dtype=np.int32)
        boardGroup = np.zeros((self.size, self.size), dtype=np.int32)

        def dfs(colorBoard, x, y, groupIndex):
            if visited[x, y] == 1:
                return
            if colorBoard[x, y] == 0:
                if self.board[x, y] == 0:
                    allLiberityPosition.add((x, y))
                return
            visited[x, y] = 1
            boardGroup[x, y] = groupIndex

            if x > 0:
                dfs(colorBoard, x - 1, y, groupIndex)
            if x < self.size - 1:
                dfs(colorBoard, x + 1, y, groupIndex)
            if y > 0:
                dfs(colorBoard, x, y - 1, groupIndex)
            if y < self.size - 1:
                dfs(colorBoard, x, y + 1, groupIndex)

        colorBoard = self.board == color

        groupIndex = 1
        for x in range(self.size):
            for y in range(self.size):
                if visited[x, y] == 0 and colorBoard[x, y] == 1:
                    allLiberityPosition = set()
                    dfs(colorBoard, x, y, groupIndex)

                    # dead group
                    if len(allLiberityPosition) == 0:
                        self.board[boardGroup == groupIndex] = 0

                    groupIndex += 1


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
