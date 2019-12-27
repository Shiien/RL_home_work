import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gym
import random
import DQN
from configTable import *
from collections import namedtuple
import os
import numpy as np
from DQN import transfor_o
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
steps_done = 0

path = os.path.abspath(os.path.dirname(__file__))
path = path + '\\Q_save_model\\'

DEBUG = False


class MyWork:
    def __init__(self):
        env = gym.envs.make("PongDeterministic-v4")
        self.Q_target = DQN.Mynet(env.observation_space, env.action_space).to(device)
        self.Q_policy = DQN.Mynet(env.observation_space, env.action_space).to(device)
        self.Q_target.load_state_dict(self.Q_policy.state_dict())
        self.Q_target.eval()
        self.env = env
        self.pool = DQN.ReplyMemory(15000)
        self.gramma = GRAMMA
        self.alpha = ALPHA
        self.epsilon = EPSILON
        self.ImageProcess = DQN.ImageProcess()

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        # eps_threshold = max(0.1, eps_threshold)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                AA = self.Q_policy(state).max(1)[1].view(1, 1)
                if DEBUG:
                    print(AA)
                return AA
        else:
            return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

    def collect(self, sample_num):
        total = 0
        total_reward = 0
        v = 0
        for i in range(100):
            state = self.env.reset()
            state = self.ImageProcess.ColorMat2Binary(state)
            state_shadow = np.stack((state, state, state, state), axis=2)
            state_now = transfor_o(state_shadow)

            while True:
                if DEBUG:
                    self.env.render()
                action = self.select_action(state_now)
                if action[0][0] == 0:
                    do_action = [[2]]
                else:
                    do_action = [[5]]
                observation1, reward, done, _ = self.env.step(do_action)
                # if done is False:
                #     observation1, reward, done, _ = self.env.step([[0]])
                next_state = self.ImageProcess.ColorMat2Binary(observation1)

                next_state = np.reshape(next_state, (80, 80, 1))
                # if DEBUG:
                #     cv2.imshow('aaa', next_state)
                next_state_shadow = np.append(next_state, state_shadow[:, :, :3], axis=2)
                if DEBUG:
                    DQN.ImageProcess.ShowImageFromNdarray(next_state_shadow)
                # if DEBUG:
                #     # for i in range(4):
                #     cv2.imshow(str(0), next_state_shadow[:][:][0])
                # for i in range(4):
                # cv2.imshow(str(0),observation1)
                state_next = transfor_o(next_state_shadow)

                total_reward += reward
                reward = torch.tensor([reward], device=device)
                self.pool.push(state_now, action,
                               state_next, reward)
                state_now = state_next
                state_shadow = next_state_shadow
                total += 1
                if done:
                    break
            v += 1
            if total >= sample_num:
                break
        print(total_reward / v)

    def sample(self):
        pass

    def update(self):
        pass

    def train(self, train_num):
        f = open('log1.txt', 'a+')
        # torch.optim.R
        opt = torch.optim.Adam(self.Q_policy.parameters())
        C = 0
        self.collect(32)
        for i in range(train_num):
            state = self.env.reset()
            state = self.ImageProcess.ColorMat2Binary(state)
            state_shadow = np.stack((state, state, state, state), axis=2)
            state_now = transfor_o(state_shadow)
            total_reward = 0
            total_loss = 0
            total_num = 0
            while True:
                if DEBUG:
                    self.env.render()
                action = self.select_action(state_now)
                do_action = None
                if action[0][0] == 0:
                    do_action = [[2]]
                else:
                    do_action = [[5]]
                observation1, reward, done, _ = self.env.step(do_action)
                # if done is False:
                #     observation1, reward, done, _ = self.env.step([[0]])
                next_state = self.ImageProcess.ColorMat2Binary(observation1)
                if DEBUG:
                    cv2.imshow('aaa', next_state)
                next_state = np.reshape(next_state, (80, 80, 1))

                next_state_shadow = np.append(next_state, state_shadow[:, :, :3], axis=2)
                state_next = transfor_o(next_state_shadow)
                reward = reward / 21.0
                total_reward += reward
                reward = torch.tensor([reward], device=device)
                self.pool.push(state_now, action,
                               state_next, reward)

                data = self.pool.sample(BATCH_SIZE)
                batch = DQN.Transition(*zip(*data))
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), device=device, dtype=torch.uint8)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                   if s is not None])
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                next_state_values[non_final_mask] = self.Q_target(non_final_next_states).max(1)[0].detach()
                reward_batch = torch.cat(batch.reward)
                expected_state_action_values = (torch.tensor(0.999) * next_state_values) + reward_batch
                k2 = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                Q = self.Q_policy(k2).gather(1, action_batch)
                loss = F.smooth_l1_loss(Q, expected_state_action_values.unsqueeze(1))
                opt.zero_grad()
                loss.backward()
                for param in self.Q_policy.parameters():
                    param.grad.data.clamp_(-1, 1)
                opt.step()
                total_loss += loss.item()
                total_num += 1
                state_now = state_next
                state_shadow = next_state_shadow
                C += 1
                if C > 100:
                    self.Q_target.load_state_dict(self.Q_policy.state_dict())
                    self.Q_target.eval()
                    C = 0
                if done:
                    break
            print(i+200, total_reward)
            print(i+200, total_loss / total_num)
            print(i+200, total_loss / total_num, file=f)

            if i % 10 == 9:
                torch.save(self.Q_policy.state_dict(),
                           path + '{}_pong_new.pt'.format(i+200))

        torch.save(self.Q_policy.state_dict(),
                   path + 'final.pt')
        torch.save(self.Q_target.state_dict(),
                   path + 'final_target.pt')
        f.close()


if __name__ == '__main__':
    print(device)
    # print(path)
    # v = list(range(1, 1000000))
    # p = map(lambda x: EPS_END + (EPS_START - EPS_END) * \
    #                   math.exp(-1. * x / EPS_DECAY), v)
    # p=list(p)
    # import matplotlib.pyplot as plt
    # plt.plot(v,p)
    # plt.show()
    # random.seed(17)
    Env = MyWork()
    Env.Q_policy.load_state_dict(torch.load(r'C:\Users\lingse\Desktop\新建文件夹\RL_home_work-master\Q_save_model\199_pong_new.pt'))
    Env.Q_target.load_state_dict(torch.load(r'C:\Users\lingse\Desktop\新建文件夹\RL_home_work-master\Q_save_model\199_pong_new.pt'))
    # Env.collect(2000)
    # while True:
    #     print(Env.alpha)
    Env.train(40000)
    # f = open('log.txt', 'a+')
    # print(10, 11, file=f)
    # print(10, 11, file=f)
    # print(10, 11, file=f)
