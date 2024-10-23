import os
import copy
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from torch.distributions.categorical import Categorical

import ais
import mlp
import utils
import vamp_utils
import samplers
import block_samplers

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

def get_sampler(args):
    data_dim = np.prod(args.input_size)
    if args.input_type == "binary":
        if args.sampler == "gibbs":
            sampler = samplers.PerDimGibbsSampler(data_dim, rand=False)
        elif args.sampler == "rand_gibbs":
            sampler = samplers.PerDimGibbsSampler(data_dim, rand=True)
        elif args.sampler.startswith("bg-"):
            block_size = int(args.sampler.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(data_dim, block_size)
        elif args.sampler.startswith("hb-"):
            block_size, hamming_dist = [int(v) for v in args.sampler.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(data_dim, block_size, hamming_dist)
        elif args.sampler == "gwg":
            sampler = samplers.DiffSampler(data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif args.sampler.startswith("gwg-"):
            n_hops = int(args.sampler.split('-')[1])
            sampler = samplers.MultiDiffSampler(data_dim, 1, approx=True, temp=2., n_samples=n_hops)
        elif args.sampler == "dmala":
            sampler = samplers.LangevinSampler(data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=True)

        elif args.sampler == "dula":
            sampler = samplers.LangevinSampler(data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.step_size, mh=False)

        
        else:
            raise ValueError("Invalid sampler...")
    else:
        if args.sampler == "gibbs":
            sampler = samplers.PerDimMetropolisSampler(data_dim, int(args.n_out), rand=False)
        elif args.sampler == "rand_gibbs":
            sampler = samplers.PerDimMetropolisSampler(data_dim, int(args.n_out), rand=True)
        elif args.sampler == "gwg":
            sampler = samplers.DiffSamplerMultiDim(data_dim, 1, approx=True, temp=2.)
        else:
            raise ValueError("invalid sampler")
    return sampler

def energy_discrepancy_bern(energy_net, samples, m_particles=32, epsilon=0.005, w_stable=1.):
    device = samples.device
    bs, dim = samples.shape

    noise_dist = dists.Bernoulli(probs=epsilon * torch.ones((dim,)).to(device))
    beri = (noise_dist.sample((bs,)) + samples) % 2.    # [bs, dim]
    pert_data = (noise_dist.sample((bs * m_particles,)).view(bs, m_particles, dim) + beri.unsqueeze(1)) % 2.    # [bs, m_particles, dim]

    pos_energy = -energy_net(samples)   # [bs]
    neg_energy = -energy_net(pert_data.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss, pos_energy.mean().item(), neg_energy.mean().item()

def energy_discrepancy_grid(energy_net, samples, m_particles=32, w_stable=1.):
    device = samples.device
    bs, dim = samples.shape

    mask = torch.zeros_like(samples)
    mask.scatter_(-1, torch.randint(dim, size=(bs,)).to(device).unsqueeze(-1), 1)
    beri = (samples + mask) % 2.    # [bs, dim]

    mask = torch.zeros(bs, m_particles, dim).to(device)
    mask.scatter_(-1, torch.randint(dim, size=(bs,m_particles,)).to(device).unsqueeze(-1), 1)
    pert_data = (beri.unsqueeze(1) + mask) % 2   # [bs, m_particles, dim]

    pos_energy = -energy_net(samples)   # [bs]
    neg_energy = -energy_net(pert_data.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss, pos_energy.mean().item(), neg_energy.mean().item()

def main(args):
    args.save_dir = os.path.join(args.save_dir, args.dataset_name, args.algo)
    utils.makedirs(args.save_dir)
    logger = open("{}/log.txt".format(args.save_dir), 'w')
    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')
        logger.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))
    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data

    # make model
    if args.model.startswith("resnet-"):
        nint = int(args.model.split('-')[1])
        net = mlp.ResNetEBM(nint)
    else:
        raise ValueError("invalid model {}".format(args.model))
    
    # get data mean and initialize buffer
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps

    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net)

    ema_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # move to cuda
    model.to(args.device)
    ema_model.to(args.device)

    # get sampler for evaluation and plotting
    sampler = get_sampler(args)

    itr = 0
    lr = args.lr
    test_ll_list = []
    best_val_ll = -np.inf
    init_dist = torch.distributions.Bernoulli(probs=init_mean.to(args.device))
    while itr < args.n_iters:
        for x in train_loader:
            if itr < args.warmup_iters:
                lr = args.lr * float(itr) / args.warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            x = preprocess(x[0].to(args.device))
            if args.algo == 'ed_grid':
                loss, pos_energy, neg_energy = energy_discrepancy_grid(model, x)
            elif args.algo == 'ed_bern':
                loss, pos_energy, neg_energy = energy_discrepancy_bern(model, x)
            else:
                raise ValueError("invalid algo {}".format(args.algo))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema_model
            for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

            if itr % args.print_every == 0:
                my_print("({}) | cur_lr = {:.8f} | pos_en = {:.8f}, "
                            "neg_en = {:.8f}, loss = {:.8f}".format(itr, lr, pos_energy, neg_energy, loss.item()))

            if (itr + 1) % args.eval_every == 0:
                logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(ema_model, init_dist, sampler,
                                                                            train_loader, val_loader, test_loader,
                                                                            preprocess, args.device,
                                                                            args.eval_sampling_steps,
                                                                            args.test_batch_size)
                my_print("EMA Train log-likelihood ({}): {}".format(itr, train_ll.item()))
                my_print("EMA Valid log-likelihood ({}): {}".format(itr, val_ll.item()))
                my_print("EMA Test log-likelihood ({}): {}".format(itr, test_ll.item()))
                test_ll_list.append(test_ll.item())
                for _i, _x in enumerate(ais_samples):
                    plot("{}/EMA_sample_{}_{}_{}_{}_{}.png".format(args.save_dir, args.dataset_name, args.sampler, args.step_size, itr, _i), _x)
            
                model.cpu()
                d = {}
                d['model'] = model.state_dict()
                d['ema_model'] = ema_model.state_dict()
                d['optimizer'] = optimizer.state_dict()
                if val_ll.item() > 0:
                    exit()
                if val_ll.item() > best_val_ll:
                    best_val_ll = val_ll.item()
                    my_print("Best valid likelihood")
                    torch.save(d, "{}/best_ckpt_{}_{}_{}.pt".format(args.save_dir,args.dataset_name,args.sampler,args.step_size))
                else:
                    torch.save(d, "{}/ckpt_{}_{}_{}.pt".format(args.save_dir,args.dataset_name,args.sampler,args.step_size))

                model.to(args.device)

            itr += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--save_dir', type=str, default="./figs/ebm")
    parser.add_argument('--algo', type=str, default='ed_grad', choices=['ed_grid', 'ed_bern'])
    parser.add_argument('--dataset_name', type=str, default='static_mnist', choices=['static_mnist', 'dynamic_mnist', 'omniglot'])

    parser.add_argument('--model', type=str, default='resnet-64')
    parser.add_argument('--base_dist', action='store_false')
    parser.add_argument('--ema', type=float, default=0.999)

    parser.add_argument('--sampler', type=str, default='gwg')
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--eval_sampling_steps', type=int, default=10000)

    parser.add_argument('--gpu', type=int, default=1, help='gpu id (default: 0)')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--warmup_iters', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=5000)

    args = parser.parse_args()
    utils.set_gpu(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
