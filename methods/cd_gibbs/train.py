import os
import torch
import shutil
import numpy as np
from tqdm import tqdm

from utils import utils
from utils.model import MLPScore, EBM
from utils.sampler import GibbsSampler
from methods.cd_gibbs.cd_loss import contrastive_divergence

def get_batch_data(db, args, batch_size=None):
    if batch_size is None:
        batch_size = args.batch_size
    bx = db.gen_batch(batch_size)
    if args.vocab_size == 2:
        bx = utils.float2bin(bx, args.bm, args.discrete_dim, args.int_scale)
    else:
        bx = utils.ourfloat2base(bx, args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    return bx

def main_loop(db, args, verbose=False):
    ckpt_path = f'{args.save_dir}/ckpts/'
    plot_path = f'{args.save_dir}/plots/'
    if os.path.exists(ckpt_path):
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)
    if os.path.exists(plot_path):
        shutil.rmtree(plot_path)
    os.makedirs(plot_path, exist_ok=True)

    net = MLPScore(args.discrete_dim*args.emb_dim, [256] * 3 + [1]).to(args.device)
    model = EBM(net, args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    sampler = GibbsSampler(args.vocab_size, args.discrete_dim, args.device)

    for epoch in range(args.num_epochs):
        model.train()
        pbar = tqdm(range(args.iter_per_epoch)) if verbose else range(args.iter_per_epoch)

        for it in pbar:
            samples = get_batch_data(db, args)
            samples = torch.from_numpy(np.float32(samples)).to(args.device)

            loss = contrastive_divergence(model, samples, sampler)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
            if verbose:
                pbar.set_description(f'Epoch {epoch} Iter {it} Loss {loss.item()}')

        if (epoch % args.epoch_save == 0) or (epoch == args.num_epochs - 1):
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

            utils.plot_heat_cat(model, db.f_scale, f'{plot_path}/heat_{epoch}.pdf', args)
            utils.plot_sampler_cat(model, f'{plot_path}/samples_{epoch}.png', args)
