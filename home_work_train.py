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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
steps_done = 0

path = os.path.abspath(os.path.dirname(__file__))
path = path + '\\Q_save_model\\'

DEBUG = False


class MyWork:
    def __init__(self):
        env = gym.envs.make("Qbert-v0")
        self.Q_target = DQN.Mynet(env.observation_space, env.action_space).to(device)
        self.Q_policy = DQN.Mynet(env.observation_space, env.action_space).to(device)
        self.Q_target.load_state_dict(self.Q_policy.state_dict())
        self.Q_target.eval()
        self.env = env
        self.pool = DQN.ReplyMemory(5000)
        self.gramma = GRAMMA
        self.alpha = ALPHA
        self.epsilon = EPSILON
        self.ImageProcess = DQN.ImageProcess()

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                AA = self.Q_policy(state).max(1)[1].view(1, 1)
                if DEBUG:
                    print(AA)
                return AA
        else:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], device=device, dtype=torch.long)

    def collect(self, sample_num):
        total = 0
        total_reward = 0
        observation = [None for _ in range(5)]
        for i in range(100):
            observation[0] = self.env.reset()
            state = self.env.reset()
            state = self.ImageProcess.ColorMat2Binary(state)
            state_shadow = np.stack((state, state, state, state), axis=0)
            state_now = transfor_o(state_shadow)
            while True:
                if DEBUG:
                    self.env.render()
                action = self.select_action(state_now)
                observation1, reward, done, _ = self.env.step(action)
                next_state = np.reshape(self.ImageProcess.ColorMat2Binary(observation1), (1, 80, 80))
                next_state_shadow = np.append(next_state, state_shadow[:3, :, :], axis=0)
                state_next = transfor_o(next_state_shadow)
                total_reward += reward
                reward = torch.tensor([reward], device=device)
                self.pool.push(state_now, action,
                               state_next, reward)
                total += 1
                state_now = state_next
                if done:
                    break
            if total >= sample_num:
                break
        print(total_reward / total)

    def sample(self):
        pass

    def update(self):
        pass

    def train(self, train_num):
        f = open('log.txt', 'a+')
        opt = torch.optim.Adam(self.Q_policy.parameters())
        for i in range(train_num):
            if i % 5 == 0:
                self.collect(BATCH_SIZE * 5)
            data = self.pool.sample(BATCH_SIZE)
            batch = DQN.Transition(*zip(*data))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.uint8)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            next_state_values[non_final_mask] = self.Q_target(non_final_next_states).max(1)[0].detach()
            reward_batch = torch.cat(batch.reward)
            expected_state_action_values = (next_state_values * self.gramma) + reward_batch
            k2 = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            Q = self.Q_policy(k2).gather(1, action_batch)
            loss = F.smooth_l1_loss(Q, expected_state_action_values.unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            for param in self.Q_policy.parameters():
                param.grad.data.clamp_(-1, 1)
            opt.step()
            print(i, loss.item())
            print(i, loss.item(), file=f)
            if i % 20 == 19:
                self.Q_target.load_state_dict(self.Q_policy.state_dict())
                self.Q_target.eval()
            if i % 500 == 499:
                torch.save(self.Q_policy.state_dict(),
                           path + '{}.pt'.format(i))

        torch.save(self.Q_policy.state_dict(),
                   path + 'final.pt')
        torch.save(self.Q_target.state_dict(),
                   path + 'final_target.pt')
        f.close()


if __name__ == '__main__':
    # print(path)
    Env = MyWork()
    # Env.collect(2000)
    # while True:
    #     print(Env.alpha)
    Env.train(1000)
    # f = open('log.txt', 'a+')
    # print(10, 11, file=f)
    # print(10, 11, file=f)
    # print(10, 11, file=f)
