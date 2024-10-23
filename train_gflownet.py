import os
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn

import utils
import vamp_utils
from mlp import ResNetEBM
from gflownet import get_GFlowNet

class EBM(nn.Module):
    def __init__(self, net, mean=None):
        super().__init__()
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            base_dist = torch.distributions.Bernoulli(probs=self.mean)
            bd = base_dist.log_prob(x).sum(-1)

        logp = -self.net(x).squeeze()
        return logp + bd

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), args.input_size[0],
        args.input_size[1], args.input_size[2]), p, normalize=True, nrow=int(x.size(0) ** .5))

    train_loader, val_loader, test_loader, args = vamp_utils.load_dataset(args)

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data
        
    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps

    net = ResNetEBM()
    model = EBM(net, init_mean)
    ckpts = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(ckpts['ema_model'])
    for param in model.parameters():
        param.requires_grad = False
    model.to(args.device)
    model.eval()

    xdim = np.prod(args.input_size)
    gfn = get_GFlowNet(args.type, xdim, args, args.device)

    itr = 0
    best_logll = -1e10
    while itr < args.n_iters:
        for x in train_loader:
            x = preprocess(x[0].to(args.device))  #  -> (bs, 784)
            
            train_loss, train_logZ = gfn.train(args.batch_size, scorer=lambda inp: model(inp).detach(),
                   itr=itr, data=x, back_ratio=args.back_ratio)

            if (itr + 1) % args.viz_every == 0:
                model.eval()
                gfn.model.eval()
                sample = gfn.sample(args.batch_size)
                plot("{}/{}_samples.png".format(args.save_dir, itr), sample)

            if (itr + 1) % args.eval_every == 0:
                model.eval()
                print("GFN TEST")
                gfn.model.eval()
                gfn_test_ll = gfn.evaluate(test_loader, preprocess, args.mc_num)
                print("GFN Test log-likelihood ({}) with {} samples: {}".format(itr, args.mc_num, gfn_test_ll.item()))

                if gfn_test_ll > best_logll:
                    best_logll = gfn_test_ll
                    gfn_ckpt = {"model": gfn.model.state_dict(), "optimizer": gfn.optimizer.state_dict(),}
                    gfn_ckpt["logZ"] = gfn.logZ.detach().cpu()
                    torch.save(gfn_ckpt, "{}/gfn_ckpt.pt".format(args.save_dir))
                    print("GFN ckpt saved!")

            itr += 1
            if itr > args.n_iters:
                print("Training finished!")
                quit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--dataset_name', type=str, default='static_mnist', choices=['static_mnist', 'dynamic_mnist', 'omniglot'])
    parser.add_argument('--ckpt_path', type=str, required=True, help="path to the pretrained EBM ckpt")

    parser.add_argument('--gpu', type=int, default=0, help='gpu id (default: 0)')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--n_iters', "--ni", type=lambda x: int(float(x)), default=5e4)
    parser.add_argument('--print_every', "--pe", type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--viz_every', type=int, default=200)

    # for GFN
    parser.add_argument("--type", type=str, default='tblb')
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--hid_layers", "--hl", type=int, default=5)
    parser.add_argument("--leaky", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gfn_bn", "--gbn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--init_zero", "--iz", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gmodel", "--gm", type=str, default="mlp")
    parser.add_argument("--train_steps", "--ts", type=int, default=1)
    parser.add_argument("--l1loss", "--l1l", type=int, default=0, choices=[0, 1], help="use soft l1 loss instead of l2")

    parser.add_argument("--with_mh", "--wm", type=int, default=1, choices=[0, 1])
    parser.add_argument("--rand_k", "--rk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lin_k", "--lk", type=int, default=1, choices=[0, 1])
    parser.add_argument("--warmup_k", "--wk", type=lambda x: int(float(x)), default=5e4, help="need to use w/ lin_k")
    parser.add_argument("--K", type=int, default=-1, help="for gfn back forth negative sample generation")

    parser.add_argument("--rand_coef", "--rc", type=float, default=0, help="for tb")
    parser.add_argument("--back_ratio", "--br", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=-1., help="for gfn's linf gradient clipping")
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--glr", type=float, default=1e-3)
    parser.add_argument("--zlr", type=float, default=1)
    parser.add_argument("--momentum", "--mom", type=float, default=0.0)
    parser.add_argument("--gfn_weight_decay", "--gwd", type=float, default=0.0)
    parser.add_argument('--mc_num', "--mcn", type=int, default=5)
    
    args = parser.parse_args()
    utils.set_gpu(args.gpu)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.save_dir = "./logs"
    main(args)

