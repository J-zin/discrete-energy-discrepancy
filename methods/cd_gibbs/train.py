import os
import torch
import shutil
import numpy as np
from tqdm import tqdm

from utils import utils
from utils.dataset import OnlineToyDataset
from methods.cd_gibbs.model import MLPScore, EBM
from methods.cd_gibbs.loss import compute_loss
from utils.dataset import plot_rings_example

from utils.sampler import MixSampler

def init_replay_buffer(args):
    buffer_size = 20000

    x_num = torch.randn(buffer_size, args.nume_size)
    x_cat = [torch.randint(0, K, (buffer_size, 1))for K in args.num_classes ]
    x_cat = torch.cat(x_cat, dim=1)
    buffer = torch.cat([x_num, x_cat], dim=1)
    return buffer

def main(args, verbose=False):
    ckpt_path = f'{args.save_dir}/ckpts/'
    plot_path = f'{args.save_dir}/plots/'
    if os.path.exists(ckpt_path):
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.makedirs(plot_path, exist_ok=True)

    dataset = OnlineToyDataset(args.dataname)
    args.nume_size = dataset.get_numerical_sizes()
    args.num_classes = dataset.get_category_sizes()

    in_dim = args.nume_size + len(args.num_classes) * args.emb_dim
    net = MLPScore(in_dim, [256] * 3 + [1]).to(args.device)
    model = EBM(net, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ema_net = MLPScore(in_dim, [256] * 3 + [1]).to(args.device)
    ema_model = EBM(ema_net, args).to(args.device)
    ema_model.load_state_dict(model.state_dict())

    sampler = MixSampler(args)

    buffer = init_replay_buffer(args)

    for epoch in range(args.num_epochs):
        pbar = tqdm(range(args.iter_per_epoch)) if verbose else range(args.iter_per_epoch)

        for it in pbar:
            samples = dataset.gen_batch(args.batch_size)
            samples = torch.from_numpy(np.float32(samples)).to(args.device)

            loss, buffer = compute_loss(model, sampler, samples, buffer)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            with torch.no_grad():
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(0.999).add_(p, alpha=0.001)

            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
            torch.save(ema_model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

            # plot buffer
            if args.dataname == 'rings':
                plot_rings_example(buffer.numpy(), f'{plot_path}/buffer_{epoch}.png')
            elif args.dataname == 'olympic':
                plot_olympic_example(buffer.numpy(), f'{plot_path}/buffer_{epoch}.png')

            # plot samples
            utils.plot_samples(ema_model, f'{plot_path}/samples_{epoch}.png', args)
