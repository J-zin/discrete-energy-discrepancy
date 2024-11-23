import torch
import numpy as np



def compute_loss(energy_net, sampler, samples, buffer, args):
    B, dim = samples.shape

    all_inds = list(range(buffer.size(0)))
    buffer_inds = sorted(np.random.choice(all_inds, B, replace=False))
    neg_samples = buffer[buffer_inds].to(sampler.device)
    
    neg_samples = sampler(energy_net, num_rounds=10, init_samples=neg_samples)
    neg_samples = neg_samples.detach()

    log_p_x_pos = -energy_net(samples)
    log_p_x_neg = -energy_net(neg_samples)

    loss_cd = torch.mean(log_p_x_neg - log_p_x_pos)
    loss_reg = torch.mean(log_p_x_pos**2 + log_p_x_neg**2)
    loss = loss_cd + 0.3 * loss_reg

    # update buffer
    buffer[buffer_inds] = neg_samples.detach().cpu()

    return loss, -log_p_x_pos.mean().item(), -log_p_x_neg.mean().item(), buffer
