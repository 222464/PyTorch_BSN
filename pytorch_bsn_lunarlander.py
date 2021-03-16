import gym
import torch
import torch.nn
from bsn.pytorch_bsn import BSN

dev = 'cpu'

env = gym.make('LunarLander-v2')

numInputs = env.observation_space.shape[0]
numActions = env.action_space.n

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(numInputs, 64, bias=True)
        self.fc2 = torch.nn.Linear(64, numActions, bias=False)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return x

model = Net().to(dev)

model_bsn = BSN(model, device=dev)

action = 0
reward = 0.0

episodeCount = 20000

for episode in range(episodeCount):
    try:
        obs = env.reset()

        totalReward = 0.0

        for t in range(1000):
            model_bsn.step(reward)

            outputs = model(torch.tensor([[[ obs.tolist() ]]], dtype=torch.float, device=dev))

            action = torch.argmax(outputs).item()

            obs, reward, done, info = env.step(action)

            totalReward += reward
            
            if done:
                print("Episode {} finished after {} timesteps, gathering {} reward.".format(episode + 1, t + 1, totalReward))

                break
        
    except KeyboardInterrupt:
        print("Stopped by user.")
        break

print("Done.")

env.close()
