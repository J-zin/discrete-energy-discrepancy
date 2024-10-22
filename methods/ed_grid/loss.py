import torch
import numpy as np

def perturb_numerical(samples, t_noise=1., m_particles=32):
    device = samples.device

    browniani = torch.randn_like(samples).to(device) * t_noise
    brownianij = torch.randn(samples.size(0), m_particles, *samples.shape[1:]).to(device) * t_noise

    pert_data = samples.unsqueeze(1) + browniani.unsqueeze(1) + brownianij
    return pert_data

def perturb_categorical(samples, args, epsilon=1., m_particles=32):
    device = samples.device
    bs, C = samples.shape # C is the number of categorical entries
    num_classes = args.num_classes

    # number of perturbed columns
    l = 1
    assert l == 1, 'Only one perturbed column is supported'

    # Per sample noise
    noise = torch.rand(bs, C, device = device)
    ids_perm = torch.argsort(noise, dim = -1)
    ids_restore = torch.argsort(ids_perm, dim = -1)

    samples_keep = torch.gather(samples, dim = -1, index = ids_perm[:, :-l]).unsqueeze(1).expand(bs, m_particles, -1)

    class_ranges = num_classes
    perturbed_ranges = torch.gather(torch.tensor(class_ranges).to(device), dim = 0, index = ids_perm[:, -l])

    uniform_noise = torch.rand(bs, m_particles, device = device)
    scaled_noise = torch.einsum('bm, b -> bm', uniform_noise, perturbed_ranges)

    samples_pert = scaled_noise.int().unsqueeze(-1)

    pert_samples = torch.cat([samples_keep, samples_pert], dim = -1)
    pert_samples = torch.gather(pert_samples, dim = -1, index = ids_restore.unsqueeze(1).expand(bs, m_particles, -1))

    return pert_samples

def compute_loss(energy_net, samples, args, epsilon = 1., m_particles = 32, w_stable = 1.):
    n_num = args.nume_size
    bs, dim = samples.shape

    x_num = samples[:, :n_num]
    x_cat = samples[:, n_num:]

    pert_x_num = perturb_numerical(x_num, t_noise=0.05*epsilon, m_particles=m_particles)
    pert_x_cat = perturb_categorical(x_cat, args, epsilon=epsilon, m_particles=m_particles)
    pert_x = torch.cat([pert_x_num, pert_x_cat], dim=-1)
    
    pos_energy = energy_net(samples)   # [bs]
    neg_energy = energy_net(pert_x.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss
