import torch
import numpy as np

def perturb_numerical(samples, t_noise=1., m_particles=32):
    device = samples.device

    browniani = torch.randn_like(samples).to(device) * t_noise
    brownianij = torch.randn(samples.size(0), m_particles, *samples.shape[1:]).to(device) * t_noise

    pert_data = samples.unsqueeze(1) + browniani.unsqueeze(1) + brownianij
    return pert_data

def perturb_categorical(samples, args, t=.1, m_particles=32):
    device = samples.device
    bs, C = samples.shape # C is the number of categorical entries
    num_classes = torch.tensor(args.num_classes, device=device)

    y = samples.clone().int()

    uniform_noise = torch.rand((bs, C), device = device)
    scaled_noise = torch.einsum('bc, c -> bc', uniform_noise, num_classes).int()
    corrupt_mask = torch.rand((bs, C)).to(samples.device) < (1 - t)
    y[corrupt_mask] = scaled_noise[corrupt_mask]

    neg_samples = y.unsqueeze(1).expand(bs, m_particles, -1).clone()
    uniform_noise = torch.rand((bs, m_particles, C), device = device)
    scaled_noise = torch.einsum('bmc, c -> bmc', uniform_noise, num_classes).int()
    corrupt_mask = torch.rand((bs, m_particles, C)).to(samples.device) < (1 - t)
    neg_samples[corrupt_mask] = scaled_noise[corrupt_mask]

    return neg_samples

def compute_loss(energy_net, samples, args, epsilon = 1., m_particles = 32, w_stable = 1.):
    n_num = args.nume_size
    bs, dim = samples.shape

    x_num = samples[:, :n_num]
    x_cat = samples[:, n_num:]

    pert_x_num = perturb_numerical(x_num, t_noise=0.05*epsilon, m_particles=m_particles)
    pert_x_cat = perturb_categorical(x_cat, args, t=0.1*epsilon, m_particles=m_particles)
    pert_x = torch.cat([pert_x_num, pert_x_cat], dim=-1)
    
    pos_energy = energy_net(samples)   # [bs]
    neg_energy = energy_net(pert_x.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss
