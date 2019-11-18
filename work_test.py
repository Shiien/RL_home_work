from time import sleep

import DQN
import torch
import gym
import random

device = 'cpu'


def transfor_o(ob):
    obb = []
    for i in ob:
        obb.append(torch.tensor(i.tolist(), dtype=torch.float32).to(device).unsqueeze(0))
    return torch.cat(obb).to(device).unsqueeze(0)


if __name__ == '__main__':
    env = gym.envs.make("PongDeterministic-v4")

    V = DQN.Mynet(env.observation_space, env.action_space)
    # V = DQN.Mynet()
    # with open('./save_model/499.pt', 'r') as f:
    V.load_state_dict(torch.load(r'C:\Users\shs\Desktop\RL_home_work\Q_save_model\25209_pong_new.pt'))
    V.eval()
    observation = [None for i in range(5)]
    import numpy as np

    # c = V(state).max(1)[1].view(1, 1)
    I = DQN.ImageProcess()

    state = env.reset()
    state = I.ColorMat2Binary(state)
    state_shadow = np.stack((state, state, state, state), axis=0)
    state_now = transfor_o(state_shadow)
    while True:
        env.render()

        print(V(state_now))
        action = V(state_now).max(1)[1].view(1, 1)
        print(action)
        if action[0][0] >= 3:
            action[0][0] = 5
        else:
            action[0][0] = 5
        observation1, reward, done, _ = env.step(action)
        next_state = np.reshape(I.ColorMat2Binary(observation1), (1, 80, 80))
        next_state_shadow = np.append(next_state, state_shadow[:3, :, :], axis=0)
        state_next = transfor_o(next_state_shadow)

        reward = torch.tensor([reward], device=device)
        state_now = state_next
        if done:
            break
