"""For all samplers in this file:
The input is a ont-hot represetaion, instead of a integer representation of the category.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
    
# Gibbs Sampler for binary and categorical data 
# (n_choices = number of categories, 2 for binary, >2 for categorical)
class GibbsSampler():
    def __init__(self, n_choices, discrete_dim, device):
        super(GibbsSampler, self).__init__()
        self.n_choices = n_choices
        self.discrete_dim = discrete_dim
        self.device = device

    def gibbs_step(self, orig_samples, axis, n_choices, score_func):
        orig_samples = orig_samples.clone()
        with torch.no_grad():
            cur_samples = orig_samples.clone().repeat(n_choices, 1)
            b = torch.LongTensor(list(range(n_choices))).to(cur_samples.device).view(-1, 1)
            b = b.repeat(1, orig_samples.shape[0]).view(-1)
            cur_samples[:, axis] = b
            score = score_func(cur_samples).view(n_choices, -1).transpose(0, 1)

            prob = F.softmax(-score, dim=-1)
            samples = torch.multinomial(prob, 1)
            orig_samples[:, axis] = samples.view(-1)
        return orig_samples

    def __call__(self, score_func, num_rounds=50, num_samples=None, init_samples=None):
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = torch.randint(self.n_choices, (num_samples, self.discrete_dim)).to(self.device)

        with torch.no_grad():
            cur_samples = init_samples.clone().float()
            for r in range(num_rounds):
                for i in range(self.discrete_dim):
                    cur_samples = self.gibbs_step(cur_samples, i, self.n_choices, score_func)

        return cur_samples.int()


# Gibbs-With-Gradients for categorical data
class GwGMultiDim():
    def __init__(self, vocab_size, discrete_dim, device, mh=True, temp=2.):
        super().__init__()
        self.discrete_dim = discrete_dim
        self.vocab_size = vocab_size
        self.device = device
        
        self.mh = mh
        self.temp = temp

    def to_one_hot(self, x):
        return F.one_hot(x.long(), self.vocab_size).float()

    def get_grad(self, x, score_func):
        x = x.float()
        x = x.requires_grad_()
        gx = torch.autograd.grad(-score_func(x).sum(), x)[0]
        return gx.detach()
    
    def diff_fn(self, x, score_func):
        x = x.float()
        x = x.requires_grad_()
        gx = torch.autograd.grad(-score_func(x).sum(), x)[0]
        gx_cur = (gx * x).sum(-1)[:, :, None]
        return gx - gx_cur

    def step(self, x, score_func):
        constant = 1.
        x_cur = self.to_one_hot(x)

        forward_delta = self.diff_fn(x_cur, score_func)
        # make sure we dont choose to stay where we are!
        forward_logits = forward_delta - constant * x_cur
        cd_forward = torch.distributions.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
        changes = cd_forward.sample()
        lp_forward = cd_forward.log_prob(changes)
        changes_r = changes.view(x_cur.size())
        changed_ind = changes_r.sum(-1)
        # mask out cuanged dim and add in the change
        x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

        if self.mh:
            reverse_delta = self.diff_fn(x_delta, score_func)
            reverse_logits = reverse_delta - constant * x_delta
            cd_reverse = torch.distributions.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (-score_func(x_delta) + score_func(x_cur))
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
        else:
            x_cur = x_delta

        return x_cur.argmax(-1)

    def __call__(self, score_func, num_rounds=400, num_samples=None, init_samples=None):
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = torch.randint(self.vocab_size, (num_samples, self.discrete_dim)).to(self.device)

        cur_samples = init_samples.clone().float()
        for r in range(num_rounds):
            cur_samples = self.step(cur_samples, score_func)

        return cur_samples.int()


# Langevin sampler for categorical data
class LangevinSamplerMultiDim():
    def __init__(self, vocab_size, discrete_dim, device, step_size=.1, mh=True, temp=2.):
        super().__init__()
        self.discrete_dim = discrete_dim
        self.vocab_size = vocab_size
        self.device = device

        self.step_size = step_size
        self.mh = mh
        self.temp = temp

    def to_one_hot(self, x):
        return F.one_hot(x.long(), self.vocab_size).float()

    def get_grad(self, x, score_func):
        x = x.float()
        x = x.requires_grad_()
        gx = torch.autograd.grad(-score_func(x).sum(), x)[0]
        return gx.detach()

    def step(self, x, score_func):
        x_cur = x
        bs = x_cur.size(0)

        x_cur_one_hot = self.to_one_hot(x)
        grad = self.get_grad(x_cur_one_hot, score_func) / self.temp
        grad_cur = grad[torch.arange(bs).unsqueeze(1), torch.arange(self.discrete_dim), x_cur.long()]
        first_term = grad.detach().clone() - grad_cur.unsqueeze(2).repeat(1, 1, self.vocab_size) 

        second_term = torch.ones_like(first_term).to(x_cur.device) / self.step_size
        second_term[torch.arange(bs).unsqueeze(1), torch.arange(self.discrete_dim), x_cur.long()] = 0
        
        cat_dist = torch.distributions.categorical.Categorical(logits=first_term-second_term)
        x_delta = cat_dist.sample()

        if self.mh:
            lp_forward = torch.sum(cat_dist.log_prob(x_delta),dim=1)
            x_delta_one_hot = self.to_one_hot(x_delta) 
            grad_delta = self.get_grad(x_delta_one_hot, score_func) / self.temp

            grad_delta_cur = grad[torch.arange(bs).unsqueeze(1), torch.arange(self.discrete_dim), x_delta.long()]
            first_term_delta = grad_delta.detach().clone() - grad_delta_cur.unsqueeze(2).repeat(1, 1, self.vocab_size)

            second_term_delta = torch.ones_like(first_term_delta).to(x_delta.device) / self.step_size
            second_term_delta[torch.arange(bs).unsqueeze(1), torch.arange(self.discrete_dim), x_delta.long()] = 0

            cat_dist_delta = torch.distributions.categorical.Categorical(logits=first_term_delta - second_term_delta)
            lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur),dim=1)

            m_term = (-score_func(x_delta_one_hot) + score_func(x_cur_one_hot))
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
        
        else:
            x_cur = x_delta
        
        return x_cur

    def __call__(self, score_func, num_rounds=400, num_samples=None, init_samples=None):
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = torch.randint(self.vocab_size, (num_samples, self.discrete_dim)).to(self.device)

        cur_samples = init_samples.clone().float()
        for r in range(num_rounds):
            cur_samples = self.step(cur_samples, score_func)

        return cur_samples.int()
