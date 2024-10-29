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
    
    # round the transition matrix to avoid numerical errors
    transition_matrix = torch.abs(transition_matrix)
    return transition_matrix


def compute_transition_matrix(args, t_noise=0.001):
    device = args.device
    qt = {}
    num_classes = args.num_classes
    cls_types = args.categorical_types

    for i, K in enumerate(num_classes):
        t = t_noise * K**2  # quadratic scaling
        # t = t_noise * K   # linear scaling
        if cls_types[i] == "ordinal":
            qt[i] = transition_matrix(t, K, ord= True).to(device)
            continue
        if cls_types[i] == "cyclical":
            qt[i] = transition_matrix(t, K, ord= False).to(device)
            continue
        else:
            qt[i] = None

        #assert torch.allclose(qt[i].sum(-1), torch.ones(K, device=device))
    return qt


def perturb_numerical(samples, t_noise=1., m_particles=32):
    device = samples.device

    browniani = torch.randn_like(samples).to(device) * t_noise
    brownianij = torch.randn(samples.size(0), m_particles, *samples.shape[1:]).to(device) * t_noise

    pert_data = samples.unsqueeze(1) + browniani.unsqueeze(1) + brownianij
    return pert_data


def perturb_cat_rate(samples, num_classes, t = 0.1, m_particles = 32):
    device = samples.device
    bs, C = samples.shape

    num_classes = torch.tensor(num_classes, device=device, dtype = torch.float32)
    if num_classes.dim() == 0:
        num_classes = num_classes.unsqueeze(0)

    #prob_stick = torch.exp(-t*num_classes)

    y = samples.clone().int()

    uniform_noise = torch.rand((bs, C), device = device)
    scaled_noise = torch.einsum('bc, c -> bc', uniform_noise, num_classes).int()  
    corrupt_mask = torch.rand((bs, C)).to(samples.device) > np.exp(-t)
    y[corrupt_mask] = scaled_noise[corrupt_mask]

    neg_samples = y.unsqueeze(1).expand(bs, m_particles, -1).clone()
    uniform_noise = torch.rand((bs, m_particles, C), device = device)
    scaled_noise = torch.einsum('bmc, c -> bmc', uniform_noise, num_classes).int()
    corrupt_mask = torch.rand((bs, m_particles, C)).to(samples.device) > np.exp(-t)
    neg_samples[corrupt_mask] = scaled_noise[corrupt_mask]

    return neg_samples


def perturb_structured(samples, K, q_t, args, m_particles=32, ord = False):

    device = samples.device
    bs = samples.shape[0]
    neg_samples = torch.zeros(bs, m_particles).to(device)

    x = F.one_hot(samples.long(), K).float()
    probs = torch.matmul(x, q_t)

    perturbation = torch.distributions.OneHotCategorical(probs)
    y = perturbation.sample()

    probs = torch.matmul(y, q_t)
    perturbation = torch.distributions.Categorical(probs)
    neg_samples[:, :] = perturbation.sample((m_particles,)).transpose(0, 1)

    return neg_samples


def perturb_categorical(samples, args, q_t, m_particles=32):
    device = samples.device
    bs, d = samples.shape

    neg_samples = torch.zeros(bs, m_particles, d).to(device)
    num_classes = args.num_classes
    cls_types = args.categorical_types
    n_uniform_cls = sum(cls_type == 'uniform' for cls_type in cls_types)

    # Uniform perturbation
    neg_samples[:, :, :n_uniform_cls] = perturb_cat_rate(samples[:, :n_uniform_cls], num_classes[:n_uniform_cls], args.tnoise, m_particles=m_particles)

    # Structured perturbation
    for i, K in enumerate(num_classes[n_uniform_cls:]):
        i = i+n_uniform_cls
        x = F.one_hot(samples[:, i].long(), K).float()
        if cls_types[i] == "cyclical":
            qt = q_t[i]
            neg_samples[:, :, i] = perturb_structured(samples[:, i], K, qt, args, m_particles=m_particles, ord = False)
            continue
        if cls_types[i] == "ordinal":
            qt = q_t[i]
            neg_samples[:, :, i] = perturb_structured(samples[:, i], K, qt, args, m_particles=m_particles, ord = True)
            continue

    return neg_samples

def compute_loss(energy_net, samples, qt_matrix, args, epsilon = 1., m_particles = 32, w_stable = 1.):
    bs, dim = samples.shape

    n_num = args.nume_size

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
