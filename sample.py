import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
from tqdm import tqdm
import torch.distributions as dists

import utils
import vamp_utils
from mlp import ResNetEBM
from gflownet import get_GFlowNet

class EBM(nn.Module):
    def __init__(self, net, mean=None):
        super().__init__()
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            base_dist = torch.distributions.Bernoulli(probs=self.mean)
            bd = base_dist.log_prob(x).sum(-1)

        logp = -self.net(x).squeeze()
        return logp + bd

def approx_difference_function(x, model):
    x = x.requires_grad_()
    gx = torch.autograd.grad(model(x).sum(), x)[0]
    wx = gx * -(2. * x - 1)
    return wx.detach()

# Gibbs-With-Gradients for binary data
class GWGSampler(nn.Module):
    def __init__(self, n_steps=1, temp=2.):
        super().__init__()
        self.n_steps = n_steps
        self.temp = temp
        self.diff_fn = lambda x, m: approx_difference_function(x, m) / self.temp


    def step(self, x, model):

        x_cur = x

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes = cd_forward.sample()

            lp_forward = cd_forward.log_prob(changes)

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur

# Discrete Langevin sampler for binary data
class LangevinSampler(nn.Module):
    def __init__(self, n_steps=1, temp=2., step_size=0.2, mh=False):
        super().__init__()
        self.n_steps = n_steps
        self.temp = temp
        self.step_size = step_size  

        self.diff_fn = lambda x, m: approx_difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []

    def step(self, x, model):

        x_cur = x
        
        EPS = 1e-10
        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            term2 = 1./(2*self.step_size) # for binary {0,1}, the L2 norm is always 1        
            flip_prob = torch.exp(forward_delta-term2)/(torch.exp(forward_delta-term2)+1)
            rr = torch.rand_like(x_cur)
            ind = (rr<flip_prob)*1
            x_delta = (1. - x_cur)*ind + x_cur * (1. - ind)

            if self.mh:
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)
                
                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta

        return x_cur

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), args.input_size[0],
        args.input_size[1], args.input_size[2]), p, normalize=True, nrow=int(x.size(0) ** .5))

    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
        
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps

    net = ResNetEBM()
    model = EBM(net, init_mean)
    ckpts = torch.load(args.ckpt_ebm, map_location='cpu')
    model.load_state_dict(ckpts['ema_model'])
    for param in model.parameters():
        param.requires_grad = False
    model.to(args.device)
    model.eval()

    xdim = np.prod(args.input_size)
    gfn = get_GFlowNet(args.type, xdim, args, args.device)

    ckpts = torch.load(args.ckpt_gfn, map_location='cpu')
    gfn.model.load_state_dict(ckpts['model'])
    gfn.model.to(args.device)
    gfn.model.eval()

    # GFlowNet sampler
    sample = gfn.sample(args.batch_size)
    plot("gfn_samples.png", sample)

    # GwG sampler
    init_dist = torch.distributions.Bernoulli(probs=init_mean.to(args.device))
    sampler = GWGSampler()
    samples = init_dist.sample((args.batch_size,))
    for _ in tqdm(range(1000)):
        samples = sampler.step(samples.detach(), model).detach()
    samples = samples.cpu()
    plot("gwg_samples.png", samples)

    # GFlowNet sampler with GWG
    sample = gfn.sample(args.batch_size)
    sampler = GWGSampler()
    for _ in tqdm(range(100)):
        sample = sampler.step(sample.detach(), model).detach()
    sample = sample.cpu()
    plot("gfn_gwg_samples.png", sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--dataset_name', type=str, default='static_mnist', choices=['static_mnist', 'dynamic_mnist'])
    parser.add_argument('--ckpt_ebm', type=str, required=True, help="path to the pretrained EBM ckpt")
    parser.add_argument('--ckpt_gfn', type=str, required=True, help="path to the pretrained GFlowNet ckpt")

    parser.add_argument('--gpu', type=int, default=1, help='gpu id (default: 0)')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--n_iters', "--ni", type=lambda x: int(float(x)), default=5e4)
    parser.add_argument('--print_every', "--pe", type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--viz_every', type=int, default=200)

    # for GFN
    parser.add_argument("--type", type=str, default='tblb')
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--hid_layers", "--hl", type=int, default=5)
    parser.add_argument("--leaky", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gfn_bn", "--gbn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--init_zero", "--iz", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gmodel", "--gm", type=str, default="mlp")
    parser.add_argument("--train_steps", "--ts", type=int, default=1)
    parser.add_argument("--l1loss", "--l1l", type=int, default=0, choices=[0, 1], help="use soft l1 loss instead of l2")

    parser.add_argument("--with_mh", "--wm", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rand_k", "--rk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lin_k", "--lk", type=int, default=1, choices=[0, 1])
    parser.add_argument("--warmup_k", "--wk", type=lambda x: int(float(x)), default=5e4, help="need to use w/ lin_k")
    parser.add_argument("--K", type=int, default=-1, help="for gfn back forth negative sample generation")

    parser.add_argument("--rand_coef", "--rc", type=float, default=0, help="for tb")
    parser.add_argument("--back_ratio", "--br", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=-1., help="for gfn's linf gradient clipping")
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--glr", type=float, default=1e-3)
    parser.add_argument("--zlr", type=float, default=1)
    parser.add_argument("--momentum", "--mom", type=float, default=0.0)
    parser.add_argument("--gfn_weight_decay", "--gwd", type=float, default=0.0)
    parser.add_argument('--mc_num', "--mcn", type=int, default=5)
    
    args = parser.parse_args()
    utils.set_gpu(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
