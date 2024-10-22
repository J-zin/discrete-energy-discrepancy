import os
import torch
import shutil
import argparse

from methods.cd_gibbs.train import main as train_cd_gibbs

from methods.ed_uni.train import main as train_ed_uni
from methods.ed_grid.train import main as train_ed_grid
from methods.ed_cyc.train import main as train_ed_cyc
from methods.ed_ord.train import main as train_ed_ord

def execute_function(method, mode):
    main_fn = eval(f'{mode}_{method}')
    return main_fn

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='rings', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='ed', help='Method: tabsyn or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--num_epochs', default=1000, type=int, help='num epochs')
    parser.add_argument('--iter_per_epoch', default=100, type=int, help='num iterations per epoch')
    parser.add_argument('--epoch_save', default=50, type=int, help='num epochs between save')

    parser.add_argument('--emb_dim', type=int, default=4, help='embedding dimension for one-hot encoding')
    parser.add_argument('--sampler', type=str, default='gibbs', choices=['gibbs', 'gwg', 'dl'])

    parser.add_argument('--cat_tnoise', default=1., type=float, help='t parameter for categorical features')

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

    args.save_dir = f'./methods/{args.method}/results/{args.dataname}'
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    return args

if __name__ == '__main__':
    args = get_args()

    main_fn = execute_function(args.method, args.mode)
    main_fn(args, verbose=True)

