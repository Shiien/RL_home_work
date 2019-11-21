import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import torch.distributions.normal as normal
import numpy as np

action_bound = None


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)


class AC(nn.Module):
    def __init__(self):
        super(AC, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )
        self.base.apply(init_weights)
        self.actor_mu = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.actor_sigma = nn.Sequential(
            nn.Linear(128, 1),
            nn.Softplus()
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.base(x)
        mu = 2 * self.actor_mu(x)
        sigma = self.actor_sigma(x) + 1e-6
        tmp = normal.Normal(mu, sigma)
        action = tmp.rsample()
        action = torch.clamp(action, min=-2, max=2)
        return tmp.log_prob(action), action, tmp.entropy(), self.critic(x)


DEBUG = False


def sample(env, net):
    state = env.reset()
    Log_p = torch.zeros((1, 1))
    en = torch.zeros((1, 1))
    R = []
    obs = []
    A = []
    while True:
        if DEBUG:
            env.render()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        log_p, ac, en, v = net(state)
        ob, r, done, _ = env.step(ac)
        state = ob
        if done:
            break
    env.close()
    return


def train(env, net, numbers):
    opt = torch.optim.Adam(net.parameters())

    for idx in range(numbers):
        S = []
        A = []
        R = []
        EN = []
        Log_p = []
        state = env.reset()
        sum_r = 0
        for gongjuren in range(300):
            if DEBUG:
                env.render()
            state = np.reshape(state, [1, 2])
            state = torch.tensor(state, dtype=torch.float32)
            S.append(state)
            # print(state)
            log_p, ac, en, v = net(state)
            # print(ac)

            Log_p.append(log_p)
            EN.append(en)
            action = ac.item()
            ob, r, done, _ = env.step([action])
            R.append(r)
            sum_r += r
            state = ob
            if done or sum_r < -200:
                break
        reward_sum = 0
        discouted_sum_reward = np.zeros_like(R)
        for t in reversed(range(0, len(R))):
            reward_sum = reward_sum * 0.995 + R[t]
            discouted_sum_reward[t] = reward_sum
        discouted_sum_reward -= np.mean(discouted_sum_reward)
        discouted_sum_reward /= np.std(discouted_sum_reward)
        # Total_loss = 0
        loss = torch.zeros(1, 1)
        for i in range(len(R)):
            loss -= Log_p[i] * (discouted_sum_reward[i]) + EN[i]
            # loss = -loss
            # Total_loss += loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(idx, reward_sum / len(R), )
    env.close()
    torch.save(net.state_dict(), './policy_1.pt')


if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    # 调用gym环境
    N = AC()
    # N.load_state_dict(torch.load(r'C:\Users\shs\Desktop\RL_home_work\policy_1.pt'))
    env = gym.make(env_name)
    # print(env.action_space,env.observation_space)
    # raise env
    action_bound = [-env.action_space.high, env.action_space.high]
    print(action_bound)
    train(env, N, 10000)
