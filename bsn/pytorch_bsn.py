import torch
import torch.nn

class BSN:
    def __init__(self, module, resolution=8, reward_dist=0.001, dtype=torch.float, device='cpu'):
        self.module = module
        self.resolution = resolution

        self.device = device

        self.params = filter(lambda p: p.requires_grad, self.module.parameters())
        self.num_params = sum(p.numel() for p in self.module.parameters() if p.requires_grad)

        self.rewards = torch.randn(self.num_params * self.resolution, dtype=dtype, device=self.device) * reward_dist
        self.divs = torch.ones(self.num_params * self.resolution, dtype=dtype, device=self.device)
        self.indices = torch.randint(low=0, high=self.resolution, size=(self.num_params,), dtype=torch.long, device=self.device)
        self.offsets = torch.tensor([ i * self.resolution for i in range(self.num_params) ], dtype=torch.long, device=self.device)

    def step(self, reward, act_scalar=1.0, alpha=0.2, epsilon=0.5):
        # Update rewards
        self.rewards[self.offsets + self.indices] += torch.divide(reward - self.rewards[self.offsets + self.indices], self.divs[self.offsets + self.indices])
        self.divs[self.offsets + self.indices] += alpha
        
        # Find new indices
        self.indices = (torch.argmax(self.rewards.reshape((self.num_params, self.resolution)), dim=1).float() + torch.randn(self.num_params, device=self.device) * epsilon + 0.5).long().clamp(min=0, max=self.resolution - 1)

        # Generate new parameters
        start_index = 0

        for p in self.module.parameters():
            p.data = (act_scalar * (self.indices[start_index : start_index + p.numel()].float() / self.resolution * 2.0 - 1.0)).reshape(p.shape)

            start_index += p.numel()
