import torch
import numpy as np
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

    def __call__(self, score_func, step_size=0.0005, num_rounds=100, num_samples=None, init_samples=None):
        """
        steps_size: step size for Langevin dynamics
        """
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = self.random_initialise(num_samples)

        cur_samples = init_samples.clone().float()
        for r in range(num_rounds):
            cur_samples = self.continuous_langevin_step(cur_samples, score_func, step_size)
            cur_samples = self.gibbs_step(cur_samples, score_func)
        
        return cur_samples

class MixLangevinSampler():
    def __init__(self, args):
        self.n_num = args.nume_size
        self.categories = args.num_classes
        self.device = args.device
        
        self.temp = 2.
        self.mh = True

    def random_initialise(self, num_samples):
        x_num = torch.randn(num_samples, self.n_num).to(self.device)
        # x_num = torch.rand(num_samples, self.n_num).to(self.device) * 4 - 2
        x_cat = [torch.randint(0, K, (num_samples, 1)).to(self.device) for K in self.categories ]
        x_cat = torch.cat(x_cat, dim=1)
        x = torch.cat([x_num, x_cat], dim=1)
        return x

    def to_one_hot(self, x):
        x_cat = x[:, self.n_num:]
        x_cat_one_hot = []
        for i in range(len(self.categories)):
            x_cat_one_hot.append(F.one_hot(x_cat[:, i].long(), self.categories[i]))
        x_cat_one_hot = torch.cat(x_cat_one_hot, dim=1)
        x = torch.cat([x[:, :self.n_num], x_cat_one_hot], dim=1)
        return x
    
    def get_grad(self, x, score_func):
        x = x.float()
        x = x.requires_grad_()
        gx = torch.autograd.grad(-score_func(x).sum(), x)[0]
        return gx.detach()

    def continuous_langevin_step(self, x, score_func, step_size):
        x = x.requires_grad_()
        x_grad = torch.autograd.grad(-score_func(x).sum(), x)[0]
        x_num_grad = x_grad[:, :self.n_num]

        x_num = x[:, :self.n_num]
        x_cat = x[:, self.n_num:]
        x_num = x_num + 0.5 * step_size * x_num_grad + np.sqrt(step_size) * torch.randn_like(x_num)
        x = torch.cat([x_num, x_cat], dim=1)
        return x.detach()
    
    def discrete_langevin_step(self, x, score_func, step_size):
        x_cur = x
        x_cur_num = x_cur[:, :self.n_num]
        x_cur_cat = x_cur[:, self.n_num:]
        bs = x_cur.size(0)

        x_cur_one_hot = self.to_one_hot(x)
        grad_all = self.get_grad(x_cur_one_hot, score_func) / self.temp

        x_delta = []
        lp_forward = []
        for i in range(len(self.categories)):
            idx_start = self.n_num + sum(self.categories[:i])
            idx_end = self.n_num + sum(self.categories[:i+1])
            grad = grad_all[:, idx_start:idx_end]
            grad_cur = grad[range(bs), x_cur_cat[:, i].long()]

            first_term = grad.detach().clone() - grad_cur.unsqueeze(1).repeat(1, self.categories[i])
            second_term = torch.ones_like(first_term).to(x_cur.device) / step_size
            second_term[range(bs), x_cur_cat[:, i].long()] = 0

            cat_dist = torch.distributions.categorical.Categorical(logits=first_term-second_term)
            x_delta_idx = cat_dist.sample()
            x_delta.append(x_delta_idx)
            lp_forward.append(cat_dist.log_prob(x_delta_idx))
        x_delta = torch.stack(x_delta, dim=1)
        lp_forward = torch.stack(lp_forward, dim=1).sum(dim=1)

        if self.mh:
            x_delta_one_hot = self.to_one_hot(torch.cat([x_cur_num, x_delta], dim=1))
            grad_delta_all = self.get_grad(x_delta_one_hot, score_func) / self.temp

            lp_reverse = []
            for i in range(len(self.categories)):
                idx_start = self.n_num + sum(self.categories[:i])
                idx_end = self.n_num + sum(self.categories[:i+1])
                grad_delta = grad_delta_all[:, idx_start:idx_end]
                grad_delta_cur = grad_delta[range(bs), x_delta[:, i].long()]

                first_term_delta = grad_delta.detach().clone() - grad_delta_cur.unsqueeze(1).repeat(1, self.categories[i])
                second_term_delta = torch.ones_like(first_term_delta).to(x_delta.device) / step_size
                second_term[range(bs), x_delta[:, i].long()] = 0

                cat_dist = torch.distributions.categorical.Categorical(logits=first_term-second_term)
                lp_reverse.append(cat_dist.log_prob(x_cur_cat[:, i]))
            lp_reverse = torch.stack(lp_reverse, dim=1).sum(dim=1)
            
            m_term = (-score_func(x_delta_one_hot) + score_func(x_cur_one_hot))
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur_cat = x_delta * a[:, None] + x_cur_cat * (1. - a[:, None])
        else:
            x_cur_cat = x_delta

        x_cur = torch.cat([x_cur_num, x_cur_cat], dim=1)

        return x_cur.detach()

    def __call__(self, score_func, step_size=0.0001, num_rounds=100, num_samples=None, init_samples=None):
        """
        steps_size: step size for Langevin dynamics
        """
        continuous_step_size = step_size
        discrete_step_size = 0.1
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = self.random_initialise(num_samples)

        cur_samples = init_samples.clone().float()
        for r in range(num_rounds):
            cur_samples = self.continuous_langevin_step(cur_samples, score_func, continuous_step_size)
            cur_samples = self.discrete_langevin_step(cur_samples, score_func, discrete_step_size)
        
        return cur_samples

class ConditionalSampler():
    """ Sampler the for categorical features conditioned on the continuous features 
    """
    def __init__(self, args):
        self.n_num = args.nume_size
        self.categories = args.num_classes
        self.device = args.device

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

    def __call__(self, score_func, num_rounds=100, init_samples=None):
        cur_samples = init_samples.clone().float()
        for r in range(num_rounds):
            cur_samples = self.gibbs_step(cur_samples, score_func)

        return cur_samples