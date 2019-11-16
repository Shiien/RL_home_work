import torch
import torch.nn as nn
import gym
import random
import torchvision.transforms as trans
import PIL.Image
from collections import namedtuple
import torchvision.models.densenet

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

magic_number = 128 * 2 * 4 * 4


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv3d:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)


class Mynet(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Mynet, self).__init__()
        assert (observation_space.shape == (210, 160, 3) and action_space.n == 4)
        self.base = nn.Sequential(
            nn.Conv3d(3, 32, (5, 3, 3), stride=2, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(32, 64, (1, 3, 3), stride=2, padding=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(64, 128, (1, 3, 3), stride=2, padding=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.V = nn.Sequential(nn.Linear(magic_number, 128), nn.ReLU(inplace=True),
                               nn.Linear(128, 1))
        self.A = nn.Sequential(nn.Linear(magic_number, 128), nn.ReLU(inplace=True),
                               nn.Linear(128, action_space.n))
        self.base.apply(init_weights)
        self.V.apply(init_weights)
        self.A.apply(init_weights)

    def forward(self, x):
        l = self.base(x)
        l = l.view(-1, magic_number)
        V = self.V(l)
        A = self.A(l)
        Q = V + (A - A.mean(dim=1, keepdim=True))
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


if __name__ == '__main__':
    env = gym.envs.make("Breakout-v0")
    a = Mynet(env.observation_space, env.action_space)

    t = env.action_space
    # while True:
    #     env.reset()
    #     while True:
    #         env.render()
    #         observation, reward, done, info = env.step([random.randint(0,3)])
    #         if done:
    #             break
    env.reset()
    observation, reward, done, info = env.step([random.randint(0, 3)])
    v = torch.tensor([[observation for i in range(5)] for j in range(10)], dtype=torch.float32)
    v = v.permute(0, 4, 1, 2, 3)

    c = a(v)
    print(c)
