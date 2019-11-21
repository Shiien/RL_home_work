import torch
import torch.nn as nn
import gym
import random
import torchvision.transforms as trans
import PIL.Image
from collections import namedtuple
import torchvision.models.densenet
import numpy as np
import cv2

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

magic_number = 128 * 2 * 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transfor_o(ob):
    obb = []

    for i in ob:
        obb.append(torch.tensor(i.tolist(), dtype=torch.float32).to(device).unsqueeze(0))
    return torch.cat(obb).to(device).unsqueeze(0)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)


class Mynet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Mynet, self).__init__()
        assert (observation_space.shape == (210, 160, 3))
        self.base = nn.Sequential(
            nn.Conv2d(4, 32, (3, 3), stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, (3, 3), stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.V = nn.Sequential(nn.Linear(magic_number, 128), nn.ReLU(inplace=True), nn.Linear(128, 128),
                               nn.ReLU(inplace=True),
                               nn.Linear(128, 1))
        self.A = nn.Sequential(nn.Linear(magic_number, 128), nn.ReLU(inplace=True), nn.Linear(128, 128),
                               nn.ReLU(inplace=True),
                               nn.Linear(128, 2))
        # 2 is up, 5 is down
        self.base.apply(init_weights)
        self.V.apply(init_weights)
        self.A.apply(init_weights)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # 调换顺序，把多通道从后面放到前面
        l = self.base(x)
        l = l.view(-1, magic_number)
        V = self.V(l)
        A = self.A(l)
        Q = V + (A - A.mean(dim=1, keepdim=True))  # dueling DQN
        return Q


class ReplyMemory(object):
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.memory = []
        self.posion = 0

    def push(self, *args):
        if len(self.memory) <= self.posion:
            self.memory.append(None)

        self.memory[self.posion] = Transition(*args)
        self.posion = (self.posion + 1) % self.maxsize

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ImageProcess():
    def ColorMat2B(self, state):
        height = 80
        width = 80
        state_gray = cv2.cvtColor(cv2.resize(state, (height, width)), cv2.COLOR_BGR2GRAY)
        _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)
        state_binarySmall = cv2.resize(state_binary, (width, height))
        cnn_inputImage = state_binarySmall.reshapeh((height, width))
        return cnn_inputImage

    def ColorMat2Binary(self, state):
        # 灰度化外加转化到80*80
        sHeight = 80
        sWidth = 80
        state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state_binarySmall = cv2.resize(state_gray, (sWidth, sHeight), interpolation=cv2.INTER_AREA)
        cnn_inputImg = state_binarySmall[25:, :]
        cnn_inputImg = cnn_inputImg.reshape((80, 80))
        return cnn_inputImg

    @staticmethod
    def ShowImageFromNdarray(state):
        # 展示图片
        pp = np.zeros((4, 80, 80), dtype=np.uint8)
        for i in range(4):
            for j in range(80):
                for k in range(80):
                    pp[i][j][k] = state[j][k][i]
        cv2.imshow(str(0), pp[0])
        cv2.imshow(str(1), pp[1])
        cv2.imshow(str(2), pp[2])
        cv2.imshow(str(3), pp[3])


if __name__ == '__main__':
    env = gym.envs.make("PongDeterministic-v4")
    a = Mynet(env.observation_space, env.action_space).cuda()

    t = env.action_space
    I = ImageProcess()
    state = env.reset()
    state = I.ColorMat2Binary(state)
    state_shadow = np.stack((state, state, state, state), axis=2)
    t = transfor_o(state_shadow).cuda()
    c = a(t)
    print(c)
