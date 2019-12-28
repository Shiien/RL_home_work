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
    V.load_state_dict(torch.load(r'C:\Users\lingse\Desktop\新建文件夹\RL_home_work-master\Q_save_model\899_pong_new.pt'))
    V.eval()
    observation = [None for i in range(5)]
    import numpy as np

    # c = V(state).max(1)[1].view(1, 1)
    I = DQN.ImageProcess()
    while True:
        state = env.reset()
        state = I.ColorMat2Binary(state)
        state_shadow = np.stack((state, state, state, state), axis=2)
        state_now = transfor_o(state_shadow)
        import time
        while True:
            env.render()
            time.sleep(0.01)
            action = V(state_now).max(1)[1].view(1, 1)
            if action[0][0] == 0:
                do_action = [[2]]
            else:
                do_action = [[5]]
            observation1, reward, done, _ = env.step(do_action)
            # if done is False:
            #     observation1, reward, done, _ = self.env.step([[0]])
            next_state = I.ColorMat2Binary(observation1)

            next_state = np.reshape(next_state, (80, 80, 1))
            # if DEBUG:
            #     cv2.imshow('aaa', next_state)
            next_state_shadow = np.append(next_state, state_shadow[:, :, :3], axis=2)
            state_next = transfor_o(next_state_shadow)
            state_now = state_next
            state_shadow = next_state_shadow
            if done:
                break
