import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class MixSampler():
    def __init__(self, args):
        self.n_num = args.nume_size
        self.categories = args.num_classes
        self.device = args.device

    def random_initialise(self, num_samples):
        x_num = torch.randn(num_samples, self.n_num).to(self.device)
        # x_num = torch.rand(num_samples, self.n_num).to(self.device) * 4 - 2
        x_cat = [torch.randint(0, K, (num_samples, 1)).to(self.device) for K in self.categories ]
        x_cat = torch.cat(x_cat, dim=1)
        x = torch.cat([x_num, x_cat], dim=1)
        return x
    
    def continuous_langevin_step(self, x, score_func, step_size):
        x = x.requires_grad_()
        x_grad = torch.autograd.grad(-score_func(x).sum(), x)[0]
        x_num_grad = x_grad[:, :self.n_num]

        x_num = x[:, :self.n_num]
        x_cat = x[:, self.n_num:]
        x_num = x_num + 0.5 * step_size * x_num_grad + np.sqrt(step_size) * torch.randn_like(x_num)
        x = torch.cat([x_num, x_cat], dim=1)
        return x.detach()
    
    @torch.no_grad()
    def gibbs_step(self, x, score_func):
        x_num = x[:, :self.n_num]
        x_cat = x[:, self.n_num:]
        for axis in range(len(self.categories)):
            K = self.categories[axis]
            cur_samples_num = x_num.clone().repeat(K, 1)
            cur_samples_cat = x_cat.clone().repeat(K, 1)
            b = torch.LongTensor(list(range(K))).to(cur_samples_cat.device).view(-1, 1)
            b = b.repeat(1, x_cat.shape[0]).view(-1)
            cur_samples_cat[:, axis] = b
            cur_samples = torch.cat([cur_samples_num, cur_samples_cat], dim=1)
            score = score_func(cur_samples).view(K, -1).transpose(0, 1)

            prob = F.softmax(-score, dim=-1)
            samples = torch.multinomial(prob, 1)
            x_cat[:, axis] = samples.view(-1)
        x = torch.cat([x_num, x_cat], dim=1)
        return x.detach()

    def __call__(self, score_func, step_size=0.0005, num_rounds=100, num_samples=None, init_samples=None, use_tqdm=False):
        """
        steps_size: step size for Langevin dynamics
        """
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = self.random_initialise(num_samples)

        tq = tqdm if use_tqdm else lambda x: x
        cur_samples = init_samples.clone().float()
        for r in tq(range(num_rounds)):
            cur_samples = self.continuous_langevin_step(cur_samples, score_func, step_size)
            cur_samples = self.gibbs_step(cur_samples, score_func)
        
        return cur_samples