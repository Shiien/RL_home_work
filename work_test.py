import DQN
import torch
import gym
import random

device = 'cpu'


def transfor_o(ob):
    obb = []
    for i in ob:
        obb.append(torch.tensor(i, dtype=torch.float32).to(device).permute(2, 0, 1).unsqueeze(0))
    return torch.cat(obb).permute(1, 0, 2, 3).to(device).unsqueeze(0)


if __name__ == '__main__':
    env = gym.envs.make("Qbert-v0")
    print(env.action_space.n)
    raise env
    V = DQN.Mynet(env.observation_space, env.action_space)
    # V = DQN.Mynet()
    # with open('./save_model/499.pt', 'r') as f:
    V.load_state_dict(torch.load(r'save_model\0.pt'))
    V.eval()
    observation = [None for i in range(5)]
    # c = V(state).max(1)[1].view(1, 1)
    while True:
        observation[0] = env.reset()
        for j in range(1, 5):
            observation[j] = observation[0]
        state_now = transfor_o(observation)
        while True:
            env.render()
            with torch.no_grad():
                # t = V(state_now).max(1)[1].view(1,1)
                action = V(state_now).max(1)[1].view(1, 1)
            # action=[random.randint(0,3)]
            observation1, reward, done, info = env.step(action[0])
            print(action)
            for j in range(0, 4):
                observation[j] = observation[j + 1]
            observation[4] = observation1
            state_next = transfor_o(observation)
            state_now = state_next
            print(done)
            if done:
                input()
                break

    print(c)
    pass
