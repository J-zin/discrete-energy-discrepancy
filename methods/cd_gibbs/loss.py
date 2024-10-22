import torch
import numpy as np

def compute_loss(energy_net, sampler, samples, buffer):
    device = samples.device
    B = samples.size(0)
    reinit_dist = torch.distributions.Bernoulli(probs=torch.tensor(0.0))

    # choose random inds from buffer
    all_inds = list(range(buffer.size(0)))
    buffer_inds = sorted(np.random.choice(all_inds, B, replace=False))
    x_buffer = buffer[buffer_inds].to(device)
    x_reinit = sampler.random_initialise(B).to(device)
    reinit = reinit_dist.sample((B,)).to(device)
    neg_samples = x_reinit * reinit[:, None] + x_buffer * (1. - reinit[:, None])


    neg_samples = sampler(energy_net, num_rounds=20, init_samples=neg_samples)
    neg_samples = neg_samples.detach()

    log_p_x_pos = -energy_net(samples)
    log_p_x_neg = -energy_net(neg_samples)
    
    loss_cd = torch.mean(log_p_x_neg - log_p_x_pos)
    loss_reg = torch.mean(log_p_x_pos**2 + log_p_x_neg**2)
    loss = loss_cd + 0.3 * loss_reg

    # update buffer
    buffer[buffer_inds] = neg_samples.detach().cpu()

    return loss, buffer