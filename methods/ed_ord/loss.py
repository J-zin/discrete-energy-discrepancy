import torch
import numpy as np
import torch.nn.functional as F

def transition_matrix(t, S, ord=False):
    """ Not currently used """
    a = torch.arange(1, S+1, 1)
    b = torch.arange(1, S+1, 1)
    p = torch.arange(0, S, 1)
    if ord:
        omega_p = np.pi*p/S
        lambda_p = 2*(torch.cos(omega_p) -1)

        in_frequency = torch.einsum('a, j -> aj', 0.5*(2*a-1), omega_p)
        out_frequency = torch.einsum('b, j -> bj', 0.5*(2*b-1), omega_p)

        in_state = torch.cos(in_frequency)*np.sqrt(2/S)
        in_state[:, 0] = 1./np.sqrt(S)
        out_state = torch.cos(out_frequency)*np.sqrt(2/S)
        out_state[:, 0] = 1./np.sqrt(S)

        amplitude = torch.exp(t*lambda_p)

        transition_matrix = torch.einsum('j, aj -> aj', amplitude, in_state)
        transition_matrix = torch.einsum('bj, aj -> ba' , out_state, transition_matrix)

    else:
        a = a - 1
        a = F.one_hot(a.long(), S).float()
        omega_p = np.pi*p/S
        lambda_p = 2*(torch.cos(2*omega_p) -1)

        in_state = torch.fft.fft(a, S, norm = "ortho")
        transition = torch.exp(t*lambda_p)*in_state
        out_state = torch.fft.ifft(transition, S, norm = "ortho")
        transition_matrix = torch.real(out_state)

    transition_matrix = torch.abs(transition_matrix)
    return transition_matrix

def compute_transition_matrix(args, t_noise=0.001, ord=False):
    device = args.device
    qt_cyc = {}
    num_classes = args.num_classes

    for i, K in enumerate(num_classes):
        t = t_noise * K**2  # quadratic scaling
        # t = t_noise * K
        qt_cyc[i] = transition_matrix(t, K, ord=ord).to(device)
        assert torch.allclose(qt_cyc[i].sum(-1), torch.ones(K, device=device))

    return qt_cyc


def perturb_numerical(samples, t_noise=1., m_particles=32):
    device = samples.device

    browniani = torch.randn_like(samples).to(device) * t_noise
    brownianij = torch.randn(samples.size(0), m_particles, *samples.shape[1:]).to(device) * t_noise

    pert_data = samples.unsqueeze(1) + browniani.unsqueeze(1) + brownianij
    return pert_data

def perturb_categorical(samples, args, q_t, m_particles=32):
    device = samples.device
    bs, d = samples.shape
    neg_samples = torch.zeros(bs, m_particles, d).to(device)

    num_classes = args.num_classes
    for i, K in enumerate(num_classes):
        x = F.one_hot(samples[:, i].long(), K).float()
        probs = torch.matmul(x, q_t[i])

        perturbation = torch.distributions.OneHotCategorical(probs)
        y = perturbation.sample()

        probs = torch.matmul(y, q_t[i])
        perturbation = torch.distributions.Categorical(probs)
        neg_samples[:, :, i] = perturbation.sample((m_particles,)).transpose(0, 1)

    return neg_samples

def compute_loss(energy_net, samples, qt_matrix, args, epsilon = 1., m_particles = 32, w_stable = 1.):
    n_num = args.nume_size
    bs, dim = samples.shape

    x_num = samples[:, :n_num]
    x_cat = samples[:, n_num:]

    pert_x_num = perturb_numerical(x_num, t_noise=0.1*epsilon, m_particles=m_particles)
    pert_x_cat = perturb_categorical(x_cat, args, qt_matrix, m_particles=m_particles)
    pert_x = torch.cat([pert_x_num, pert_x_cat], dim=-1)
    
    pos_energy = energy_net(samples)   # [bs]
    neg_energy = energy_net(pert_x.view(-1, dim)).view(bs, -1)  # [bs, m_particles]
    val = pos_energy.view(bs, 1) - neg_energy
    if w_stable != 0:
        val = torch.cat([val, np.log(w_stable) * torch.ones_like(val[:, :1])], dim=-1)
    
    loss = val.logsumexp(dim=-1).mean()
    return loss
