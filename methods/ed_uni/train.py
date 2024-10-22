import os
import torch
import shutil
import numpy as np
from tqdm import tqdm

from utils import utils
from utils.dataset import OnlineToyDataset
from methods.ed_uni.model import MLPScore, EBM
from methods.ed_uni.loss import compute_loss


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

    for epoch in range(args.num_epochs):
        pbar = tqdm(range(args.iter_per_epoch)) if verbose else range(args.iter_per_epoch)

        for it in pbar:
            samples = dataset.gen_batch(args.batch_size)
            samples = torch.from_numpy(np.float32(samples)).to(args.device)

            loss = compute_loss(model, samples, args)

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
        # if True:
            torch.save(ema_model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')
            utils.plot_samples(ema_model, f'{plot_path}/samples_{epoch}.png', args)
