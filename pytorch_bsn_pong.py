import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from bsn.pytorch_bsn import BSN
import numpy as np
import lycon

imageSize = ( 84, 84 )

env = gym.make('Pong-v0')

minSize = min(env.observation_space.shape[0], env.observation_space.shape[1])
maxSize = max(env.observation_space.shape[0], env.observation_space.shape[1])

class Net(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc4 = nn.Linear(7 * 7 * 64, 512, bias=False)
        self.fc5 = nn.Linear(512, num_actions, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))

        return self.fc5(x)

env = gym.make('Pong-v0')

model = Net(1, env.action_space.n)

model_bsn = None

action = 0
reward = 0.0

episodeCount = 20000

for episode in range(episodeCount):
    try:
        obs = env.reset()

        for t in range(1000):
            env.render()
            
            obs = obs.astype(dtype=np.float32) / 255.0

            obs = obs[maxSize // 2 - minSize // 2 : maxSize // 2 + minSize // 2, :, :]

            obs = (obs[:, :, 0] + obs[:, :, 1] + obs[:, :, 2]) / 3.0

            obs = lycon.resize(obs, width=imageSize[0], height=imageSize[1], interpolation=lycon.Interpolation.CUBIC)

            action = model(torch.tensor(list(obs.reshape((1, 1, obs.shape[0], obs.shape[1])))))

            if model_bsn is None:
                model_bsn = BSN(model)

            action = int(torch.argmax(action))

            obs, _, done, info = env.step(action)
            
            reward = 0.0

            if done:
                reward = -1.0
            
            model_bsn.step(reward)

            if done:
                print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))

                break
        
    except KeyboardInterrupt:
        print("Stopped by user.")
        break

print("Done.")

env.close()