import torch
import torch.nn

class BSN:
    def __init__(self, module, resolution=64, reward_dist=0.001, dtype=torch.float):
        self.module = module
        self.resolution = resolution

        self.params = filter(lambda p: p.requires_grad, self.module.parameters())
        self.num_params = sum(p.numel() for p in self.module.parameters() if p.requires_grad)

        self.rewards = torch.randn(self.num_params * self.resolution, dtype=dtype) * reward_dist
        self.indices = torch.randint(low=0, high=self.resolution, size=(self.num_params,), dtype=torch.long)
        self.offsets = torch.tensor([ i * self.resolution for i in range(self.num_params) ], dtype=torch.long)

    def step(self, reward, act_scalar=4.0, alpha=0.001, epsilon=0.3):
        # Update rewards
        self.rewards[self.offsets + self.indices] += alpha * (reward - self.rewards[self.offsets + self.indices])
        
        # Find new indices
        self.indices = (torch.argmax(self.rewards.reshape((self.num_params, self.resolution)), dim=1).float() + torch.randn(self.num_params) * epsilon + 0.5).long().clamp(min=0, max=self.resolution - 1)

        # Generate new parameters
        start_index = 0

        for p in self.module.parameters():
            p.data = (act_scalar * (self.indices[start_index : start_index + p.numel()].float() / self.resolution * 2.0 - 1.0)).reshape(p.shape)

            start_index += p.numel()