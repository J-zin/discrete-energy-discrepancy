import torch
import torch.distributions as dists
import numpy as np

def perturb_cat_grid(samples, num_classes, m_particles=32):
    device = samples.device
    bs, C = samples.shape # C is the number of categorical entries

    # Per sample noise
    noise = torch.rand(bs, C, device = device)
    ids_perm = torch.argsort(noise, dim = -1)
    ids_restore = torch.argsort(ids_perm, dim = -1)

    samples_keep = torch.gather(samples, dim = -1, index = ids_perm[:, :-1]).unsqueeze(1).expand(bs, m_particles, -1)
    pert_num_classes = torch.gather(num_classes, dim = 0, index = ids_perm[:, -1])

    uniform_noise = torch.rand(bs, m_particles, device = device)
    scaled_noise = torch.einsum('bm, b -> bm', uniform_noise, pert_num_classes)
    samples_pert = scaled_noise.int().unsqueeze(-1)

    pert_samples = torch.cat([samples_keep, samples_pert], dim = -1)
    pert_samples = torch.gather(pert_samples, dim = -1, index = ids_restore.unsqueeze(1).expand(bs, m_particles, -1))
    return pert_samples

def ed_categorical(energy_net, samples, K=5, dim=16, m_particles=32, w_stable=1.):
    device = samples.device
    bs, dim = samples.shape

    num_classes = torch.tensor([K] * dim, device=device)
    neg_data = perturb_cat_grid(samples, num_classes, m_particles)   # [bs, m_particles, dim]

    pos_energy = energy_net(samples)   # [bs]
    neg_energy = energy_net(neg_data.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss

def ed_binary(energy_net, samples, epsilon=0.1, m_particles=32, w_stable=1.):
    device = samples.device
    bs, dim = samples.shape

    noise_dist = dists.Bernoulli(probs=epsilon * torch.ones((dim,)).to(device))
    beri = (noise_dist.sample((bs,)) + samples) % 2.    # [bs, dim]
    pert_data = (noise_dist.sample((bs * m_particles,)).view(bs, m_particles, dim) + beri.unsqueeze(1)) % 2.    # [bs, m_particles, dim]

    pos_energy = energy_net(samples)   # [bs]
    neg_energy = energy_net(pert_data.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss
