import os
import torch
import argparse
import torchvision
import numpy as np

import ais
import mlp
import utils
import vamp_utils
from train_ed import get_sampler, EBM

"""
python -u eval_ais.py \
    --dataset_name static_mnist \
    --algo ed_grid \
    --sampler gwg \
    --step_size 0.1 \
    --sampling_steps 40 \
    --model resnet-64 \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --viz_every 1000 \
    --gpu 0
"""

def main(args):
    args.save_dir = os.path.join(args.save_dir, args.dataset_name, args.algo)
    logger = open("{}/eval_ais_log.txt".format(args.save_dir), 'w')

    def my_print(s):
        print(s)
        logger.write(str(s) + '\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    my_print("Loading data")
    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0),
                                                            args.input_size[0], args.input_size[1], args.input_size[2]),
                                                     p, normalize=True, nrow=int(x.size(0) ** .5))

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
        
    # make model
    my_print("Making Model")
    if args.model.startswith("resnet-"):
        nint = int(args.model.split('-')[1])
        net = mlp.ResNetEBM(nint)
    else:
        raise ValueError("invalid model {}".format(args.model))
    
    # get data mean and initialize buffer
    my_print("Getting init batch")
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps

    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net)

    d = torch.load("{}/best_ckpt_{}_{}_{}.pt".format(args.save_dir,args.dataset_name,args.sampler,args.step_size), map_location='cpu')

    if args.ema:
        model.load_state_dict(d['ema_model'])
    else:
        model.load_state_dict(d['model'])

    # wrap model for annealing
    init_dist = torch.distributions.Bernoulli(probs=init_mean.to(args.device))

    # get sampler
    sampler = get_sampler(args)

    my_print(args.device)
    my_print(model)
    my_print(sampler)

    logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(model, init_dist, sampler,
                                                                train_loader, val_loader, test_loader,
                                                                preprocess, args.device,
                                                                args.eval_sampling_steps,
                                                                args.n_samples, viz_every=args.viz_every)
    my_print("EMA Train log-likelihood: {}".format(train_ll.item()))
    my_print("EMA Valid log-likelihood: {}".format(val_ll.item()))
    my_print("EMA Test log-likelihood: {}".format(test_ll.item()))
    for _i, _x in enumerate(ais_samples):
        plot("{}/ais_EMA_sample_{}_{}_{}_{}.png".format(args.save_dir,args.dataset_name, args.sampler,args.step_size,_i), _x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--save_dir', type=str, default="./figs/ebm")
    parser.add_argument('--algo', type=str, default='ed_grad', choices=['ed_grid', 'ed_bern'])
    parser.add_argument('--dataset_name', type=str, default='static_mnist', choices=['static_mnist', 'dynamic_mnist', 'omniglot', 'caltech'])

    parser.add_argument('--model', type=str, default='resnet-64')
    parser.add_argument('--base_dist', action='store_false')
    parser.add_argument('--ema', type=float, default=0.999)

    parser.add_argument('--sampler', type=str, default='dmala')
    parser.add_argument('--sampling_steps', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--eval_sampling_steps', type=int, default=10000)

    parser.add_argument('--gpu', type=int, default=0, help='gpu id (default: 0)')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--n_iters', type=int, default=100000)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--warmup_iters', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=.0)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=5000)

    args = parser.parse_args()
    utils.set_gpu(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)
