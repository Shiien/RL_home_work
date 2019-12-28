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
        # if DEBUG:
        #     action = mu
        # else:
        action = tmp.rsample()
        action = torch.clamp(action, min=-2, max=2)
        return tmp.log_prob(action), action, tmp.entropy(), self.critic(x)


DEBUG = False


def sample(env, net):
    state = env.reset()
    Log_p = []
    En = []
    R = []
    V = []

    for gongjuren in range(300):
        if DEBUG:
            env.render()
        state = np.reshape(state, [1, 2])
        state = torch.tensor(state, dtype=torch.float32)
        log_p, ac, en, v = net(state)
        V.append(v)
        Log_p.append(log_p)
        En.append(en)
        ob, r, done, _ = env.step([ac.item()])
        R.append(r)
        state = ob
        if done:
            break
    reward_sum = 0
    discouted_sum_reward = np.zeros_like(R)
    # Log_p = torch.tensor(Log_p, dtype=torch.float32)
    # V = torch.tensor(V, dtype=torch.float32)
    # R = torch.tensor(R, dtype=torch.float32)
    # En = torch.tensor(En, dtype=torch.float32)

    for t in reversed(range(0, len(R))):
        reward_sum = reward_sum * 0.995 + R[t]
        discouted_sum_reward[t] = float(reward_sum)
    # discouted_sum_reward -= np.mean(discouted_sum_reward)
    # discouted_sum_reward /= np.std(discouted_sum_reward)
    # discouted_sum_reward = torch.tensor(discouted_sum_reward, dtype=torch.float32).squeeze()
    loss = torch.zeros(1, 1)
    for i in range(len(R)):
        loss -= (discouted_sum_reward[i]-V[i].item())*Log_p[i]
    # loss = loss / len(R)
    loss1 = torch.zeros(1, 1)
    for i in range(len(R)):
        loss1 += (V[i] - discouted_sum_reward[i]) * (V[i] - discouted_sum_reward[i])
    # loss1 /= len(R)
    return loss, reward_sum, loss1


def train(env, net, numbers):
    opt = torch.optim.Adam(net.parameters())

    for idx in range(numbers):
        sum_r = 0
        Loss = torch.zeros(1, 1)
        Loss1 = torch.zeros(1, 1)
        for gongjuren in range(50):
            loss, r, loss1 = sample(env, net)
            Loss += loss
            Loss1 += loss1
            sum_r += r
        # Loss = torch.cat(Loss)
        total_loss = (Loss + Loss1) / 50
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        print(idx, sum_r / 50)
        if idx % 20 == 19:
            torch.save(net.state_dict(), './policy_{}.pt'.format(idx +4100))
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
    N.load_state_dict(torch.load(r'C:\Users\93569\Desktop\RL_home_work\policy_4099.pt'))
    # action_bound = [-env.action_space.high, env.action_space.high]
    # print(action_bound)
    # train(env, N, 10000)
    DEBUG=True
    sample(env,N)
    env.close()