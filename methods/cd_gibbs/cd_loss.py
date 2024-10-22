import torch

def contrastive_divergence(energy_net, samples, sampler):
    neg_samples = samples.clone()
    neg_samples = sampler(energy_net, num_rounds=10, init_samples=neg_samples)
    neg_samples = neg_samples.detach()

    log_p_x_pos = -energy_net(samples)
    log_p_x_neg = -energy_net(neg_samples)
    
    loss_cd = torch.mean(log_p_x_neg - log_p_x_pos)
    loss_reg = torch.mean(log_p_x_pos**2 + log_p_x_neg**2)
    loss = loss_cd + 0.3 * loss_reg
    return loss