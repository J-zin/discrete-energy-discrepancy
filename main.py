import os
import torch
import shutil
import argparse
import numpy as np

from utils import utils
from methods.ed_grid.train import main_loop as ed_grid_main_loop
from methods.cd_gibbs.train import main_loop as cd_gibbs_main_loop


def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    parser.add_argument('--methods', type=str, default='punidb', 
        choices=[
            'cd_gibbs','ed_grid'
        ],
    )

    parser.add_argument('--data_name', type=str, default='moons')
    parser.add_argument('--discrete_dim', type=int, default=16)
    parser.add_argument('--vocab_size', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=4, help='embedding dimension for one-hot encoding')

    parser.add_argument('--gpu', type=int, default=0, help='-1: cpu; 0 - ?: specific gpu index')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--sampler', type=str, default='gibbs', choices=['gibbs', 'gwg', 'dmla'])

    parser.add_argument('--num_epochs', default=1000, type=int, help='num epochs')
    parser.add_argument('--iter_per_epoch', default=100, type=int, help='num iterations per epoch')
    parser.add_argument('--epoch_save', default=100, type=int, help='num epochs between save')

    args = parser.parse_args()

    gpu = args.gpu
    if gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        args.device = torch.device('cuda:' + str(gpu))
        print('use gpu indexed: %d' % gpu)
    else:
        args.device = torch.device('cpu')
        print('use cpu')

    args.save_dir = f'./methods/{args.methods}/results/voc_size={args.vocab_size}/{args.data_name}'
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    return args

def plot_binary_data_samples(db, args):
    data = utils.float2bin(db.gen_batch(1000), args.bm, args.discrete_dim, args.int_scale)
    float_data = utils.bin2float(data.astype(np.int32), args.inv_bm, args.discrete_dim, args.int_scale)
    utils.plot_samples(float_data, f'{args.save_dir}/data_sample.pdf', im_size=4.1, im_fmt='pdf')

def plot_categorical_data_samples(db, args):
    data = utils.ourfloat2base(db.gen_batch(1000), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    float_data = utils.ourbase2float(data.astype(np.int32), args.discrete_dim, args.f_scale, args.int_scale, args.vocab_size)
    utils.plot_samples(float_data, f'{args.save_dir}/data_sample.pdf', im_size=4.1, im_fmt='pdf')

if __name__ == '__main__':
    args = get_args()
    if args.vocab_size == 2:
        args.discrete_dim = 32
        db, bm, inv_bm = utils.setup_data(args)
        args.bm = bm
        args.inv_bm = inv_bm
        plot_binary_data_samples(db, args)
    else:
        db = utils.our_setup_data(args)
        plot_categorical_data_samples(db, args)

    main_fn = eval(f'{args.methods}_main_loop')

    main_fn(db, args, verbose=True)
