import numpy as np
from tqdm import tqdm

from utils.sampler import MixSampler, MixLangevinSampler
from utils.dataset import plot_rings_example

def plot_samples(score_func, out_file, args):
    if args.sampler == 'gibbs':
        sampler = MixSampler(args)
    elif args.sampler == 'dl':
        sampler = MixLangevinSampler(args)
    else:
        raise ValueError(f'Unknown sampler {args.sampler}')

    if args.dataname == 'rings':
        steps_size = 0.0001
        num_rounds = 1000
    else:
        raise ValueError(f'Unknown dataname {args.dataname}')

    samples = []
    for _ in tqdm(range(10)):
        gen_samples = sampler(score_func, steps_size, num_rounds=num_rounds, num_samples=2000)
        samples.append(gen_samples.data.cpu().numpy())
    samples = np.concatenate(samples, axis=0)

    if args.dataname == 'rings':
        plot_rings_example(samples, out_file)
    else:
        raise ValueError(f'Unknown dataname {args.dataname}')
