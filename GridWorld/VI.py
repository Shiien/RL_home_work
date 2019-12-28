import random
import numpy as np


class Map:
    def __init__(self, eps=0, gamma=0.95):
        self.M = ["*************************",
                  "*      *    *        **X*",
                  "*      *    *        ** *",
                  "*S     *            *** *",
                  "*      ******           *",
                  "*                      **",
                  "* **  **  ** ***  *******",
                  "*************************"]
        self.x = 3
        self.y = 1
        self.Xmax = len(self.M)
        self.Ymax = len(self.M[0])
        self.Pi = [[0 for _ in range(self.Ymax)] for __ in range(self.Xmax)]
        self.V = [[0.0 for _ in range(self.Ymax)] for __ in range(self.Xmax)]
        self.eps = eps
        self.gamma = gamma

    def step(self, action):
        # print(action)
        assert action >= 0 and action <= 3 and isinstance(action, int)
        if action == 0:
            self.x -= 1
        elif action == 1:
            self.x += 1
        elif action == 2:
            self.y -= 1
        elif action == 3:
            self.y += 1
        if self.x <= 0:
            self.x = 0
        if self.y <= 0:
            self.y = 0
        if self.x >= self.Xmax:
            self.x = self.Xmax - 1
        if self.y >= self.Ymax:
            self.y = self.Ymax - 1
        if self.M[self.x][self.y] == 'X':
            return (self.x, self.y), 100, True, 0
        if self.M[self.x][self.y] == ' ' or self.M[self.x][self.y] == 'S':
            return (self.x, self.y), -1, False, 0
        if self.M[self.x][self.y] == '*':
            return (self.x, self.y), -100, True, 0

    def reset(self):
        self.x = 3
        self.y = 1

    def reward(self, x, y):
        if x < 0 or x >= self.Xmax or y < 0 or y >= self.Ymax:
            return -100
        if self.M[x][y] == 'X':
            return 100
        if self.M[x][y] == ' ' or self.M[x][y] == 'S':
            return -1
        if self.M[x][y] == '*':
            return -100

    def getV(self, x, y):
        if x < 0 or x >= self.Xmax:
            return -100
        if y < 0 or y >= self.Ymax:
            return -100
        return self.V[x][y]

    def getReturn(self, x, y, action):
        if action == 0:
            x = x - 1
        if action == 1:
            x = x + 1
        if action == 2:
            y = y - 1
        if action == 3:
            y = y + 1
        r = self.reward(x, y)
        v = self.getV(x, y)
        return r + self.gamma * v

    def V_E(self):
        while True:
            delta = 0.0
            for i in range(1, self.Xmax - 1):
                for j in range(1, self.Ymax - 1):
                    if self.M[i][j] in '*X':
                        continue
                    R = self.getReturn(i, j, self.Pi[i][j]) * 0.95
                    for k in range(4):
                        R += self.getReturn(i, j, k) * 0.05 / 4
                    delta += abs(R - self.V[i][j])
                    self.V[i][j] = R
            if self.eps == 0 or delta < self.eps:
                return

    def V_I(self):
        for i in range(1, self.Xmax - 1):
            for j in range(1, self.Ymax - 1):
                if self.M[i][j] in "X*":
                    continue
                L = []
                for k in range(4):
                    L.append(self.getReturn(i, j, k))
                ne = np.argmax(L)
                self.Pi[i][j] = int(ne)

    def save(self):
        import json
        with open('./save_VIVI.json', 'w') as f:
            json.dump(self.Pi, f)


if __name__ == '__main__':
    M = Map()
    for i in range(10000):
        M.V_E()
        M.V_I()
    M.save()
    M.reset()
    while True:
        action = M.Pi[M.x][M.y]
        L = [M.getReturn(M.x, M.y, k) for k in range(4)]
        print(L)
        ob, r, done, _ = M.step(action)
        print(ob, r, done)
        if done:
            break
