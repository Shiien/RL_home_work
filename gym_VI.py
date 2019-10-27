import gym
import numpy as np
import random

env = gym.make("MountainCarContinuous-v0")
print(env.action_space, env.observation_space, env.reward_range)

lidu = 100


class Value:
    def __init__(self, alpha=0.2, epsilon=0.8, beta=0.99):
        self.Q = [[0.0 for _ in range(2 * lidu+1)] for __ in range(4*lidu * lidu+1)]
        self.Q = np.array(self.Q)
        self.M = []
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta

    def pick_action(self, observation):
        S = Value.observation_to_S(observation)
        A = np.argmax(self.Q[S])
        #print(A)

        return Value.A_to_action(A)

    def reset_memory(self):
        self.M = []

    def push_memory(self, Ob1, ac1, Ob2, R):
        o1 = Value.observation_to_S(Ob1)
        o2 = Value.observation_to_S(Ob2)
        a = Value.action_to_A(ac1)
        self.M.append([o1, a, o2, R])

    def update(self):
        for i in self.M:
            o1, a, o2, r = i
            self.Q[o1, a] = self.Q[o1, a] + self.alpha * (
                    r + self.beta * np.amax(self.Q[o2]) - self.Q[o1, a])
            # print(self.Q[o1,a])

    @staticmethod
    def observation_to_S(observation):
        p = int((observation[0] + 1) / (1 / lidu))
        #print(p)
        # v = int(observation[1] * lidu) + lidu
        v = int((observation[1] + 1) / (1 / lidu))
        return p * 2*lidu + v

    @staticmethod
    def A_to_action(A):
        return [A * (1 / 2*lidu) - 1 + random.random() / lidu]
        # return [(A - lidu) / lidu + random.random() / lidu]

    @staticmethod
    def action_to_A(action):
        return int((action[0] + 1) / (1 / 2*lidu))
        # return int(action[0] * lidu + lidu)


N = Value()
# for i in range(2*lidu):
#     print(N.A_to_action(i),N.action_to_A(N.A_to_action(i)))
# raise N
for ep in range(10000):
    N.reset_memory()
    observation = env.reset()
    R = []
    for _ in range(1000):
        env.render()
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        if random.random() < N.epsilon:
            action = env.action_space.sample()
        else:
            action = N.pick_action(observation)
        # print(action)
        # action = env.action_space.sample()
        # print(action)
        obf = observation
        observation, reward, done, info = env.step(action)
        N.push_memory(obf, action, observation, reward)
        N.update()
        N.reset_memory()
        # print(N.M[-1])
        R.append(reward)
        # print(observation, reward, info)
        if done:
            observation = env.reset()
        N.epsilon = N.epsilon * 0.95
        if N.epsilon<0.1:
            N.epsilon=0.1
    print(max(R))
    R = []
    N.update()
env.close()
