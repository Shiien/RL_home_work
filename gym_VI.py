import gym
import numpy as np
import random
'''
you should run command 
'> pip install gym'
'> pip install numpy'
'> pip install json'
before
'''

lidu = 100


class Value:
    def __init__(self, env, policy='on', mode='TD', visit=True, alpha=0.2, epsilon=0.8, beta=0.99):
        self.Q = [[0.0 for _ in range(2 * lidu + 1)] for __ in range(4 * lidu * lidu + 1)]
        self.Q = np.array(self.Q)
        self.M = []
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.env = env
        self.visit = visit
        self.mode = mode
        self.n = None
        if self.mode == 'MC':
            self.n = [[0.001 for _ in range(2 * lidu + 1)] for __ in range(4 * lidu * lidu + 1)]
        self.policy = policy

    def pick_action_greedy(self, observation):
        s = Value.observation_to_S(observation)
        return Value.A_to_action(np.argmax(self.Q[s]))

    def pick_action(self, observation):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        if type(observation) is not int:
            S = Value.observation_to_S(observation)
        else:
            S = observation
        A = np.argmax(self.Q[S])
        # print(A)

        return Value.A_to_action(A)

    def reset_memory(self):
        self.M = []

    def push_memory(self, Ob1, ac1, Ob2, R):
        o1 = Value.observation_to_S(Ob1)
        o2 = Value.observation_to_S(Ob2)
        a = Value.action_to_A(ac1)
        self.M.append([o1, a, o2, R])

    def update(self):
        if self.mode is 'MC':
            g = self.Q[self.M[-1][2], Value.action_to_A(self.pick_action(self.M[-1][2]))]
            for i in range(len(self.M, -1, -1)):
                g *= self.beta
                g += self.M[i][3]

            for i in range(len(self.M)):
                o1, a, o2, r = i
                self.n[o1, a] += 1.0
                self.Q[o1, a] = (self.Q[o1, a] * self.n[o1, a] - 1 + g) / self.n[o1, a]
                g -= r
                g /= self.beta
        else:
            for i in self.M:
                o1, a, o2, r = i
                if self.mode is 'off':
                    self.Q[o1, a] = self.Q[o1, a] + self.alpha * (
                            r + self.beta * np.amax(self.Q[o2]) - self.Q[o1, a])
                else:
                    self.Q[o1, a] = self.Q[o1, a] + self.alpha * (
                            r + self.beta * self.Q[o2, Value.action_to_A(self.pick_action(o2))] - self.Q[o1, a])
            # print(self.Q[o1,a])

    @staticmethod
    def observation_to_S(observation):
        p = int((observation[0] + 1) / (1 / lidu))
        # print(p)
        # v = int(observation[1] * lidu) + lidu
        v = int((observation[1] + 1) / (1 / lidu))
        return p * 2 * lidu + v

    @staticmethod
    def A_to_action(A):
        return [A * (1 / 2 * lidu) - 1 + random.random() / lidu]
        # return [(A - lidu) / lidu + random.random() / lidu]

    @staticmethod
    def action_to_A(action):
        return int((action[0] + 1) / (1 / 2 * lidu))
        # return int(action[0] * lidu + lidu)

    def step(self):
        self.epsilon *= 0.95
        if self.epsilon < 0.1:
            self.epsilon = 0.1

    def train(self, eps=1000):
        for ep in range(eps):
            observation = self.env.reset()
            for _ in range(100):
                if self.visit is True:
                    self.env.render()
                action = self.pick_action(observation)
                obf = observation
                observation, reward, done, info = self.env.step(action)
                N.push_memory(obf, action, observation, reward)
                if self.mode is 'TD':
                    self.update()
                    self.reset_memory()
                if done:
                    break
            self.update()
            self.reset_memory()
            self.step()

    def save_Q(self):
        self.env.close()
        import json
        with open('./save1.json', 'w') as f:
            json.dump(self.Q.tolist(), f)


N = Value(gym.make("MountainCarContinuous-v0"), policy='MC')
N.train(1)

N.save_Q()
