import gym
import torch
import torch.nn
import torch.nn.functional as F
from bsn.pytorch_bsn import BSN

env = gym.make('CartPole-v0')

numInputs = env.observation_space.shape[0]

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(numInputs, 16, bias=False)
        self.fc2 = torch.nn.Linear(16, 1, bias=False)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)

        return x

env = gym.make('CartPole-v0')

model = Net()

model_bsn = BSN(model)

action = 0
reward = 0.0

episodeCount = 20000

for episode in range(episodeCount):
    try:
        obs = env.reset()

        for t in range(1000):
            model_bsn.step(reward)
            action = model(torch.tensor([[[list(obs)]]]))

            if action > 0.0:
                action = 1
            else:
                action = 0

            obs, _, done, info = env.step(action)
            
            reward = 0.0

            if done:
                reward = -1.0

            if done:
                print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))

                break
        
    except KeyboardInterrupt:
        print("Stopped by user.")
        break

print("Done.")

env.close()