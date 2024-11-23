import os
import time
import torch
import numpy as np
import pandas as pd

from copy import deepcopy

import src
from utils_train import make_dataset
from methods.cd_gibbs.models.model import MLPScore, EBM
from methods.cd_gibbs.models.loss import compute_loss
from methods.cd_gibbs.models.sampler import MixSampler

def init_replay_buffer(args):
    buffer_size = 10000

    x_num = torch.randn(buffer_size, args.nume_size)
        # x_num = torch.rand(num_samples, self.n_num).to(self.device) * 4 - 2
    x_cat = [torch.randint(0, K, (buffer_size, 1))for K in args.num_classes]
    x_cat = torch.cat(x_cat, dim=1)
    buffer = torch.cat([x_num, x_cat], dim=1)
    return buffer

class Trainer:
    def __init__(self, model, train_iter, lr, weight_decay, steps, model_save_path, device=torch.device('cuda:1'), args=None):
        self.model = model
        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'loss'])
        self.model_save_path = model_save_path

        columns = list(np.arange(5)*200)
        columns[0] = 1
        columns = ['step'] + columns
 
        self.args = args

        self.log_every = 50
        self.print_every = 1

        self.buffer = init_replay_buffer(args)
        self.sampler = MixSampler(args)

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x):
        x = x.to(self.device)

        self.optimizer.zero_grad()

        loss, pos_en, neg_en, self.buffer = compute_loss(self.model, self.sampler, x, self.buffer, self.args)

        loss.backward()
        self.optimizer.step()

        return loss, pos_en, neg_en


    def run_loop(self):
        step = 0
        curr_loss = 0.0

        curr_count = 0
        self.print_every = 1
        self.log_every = 1

        best_loss = np.inf
        print('Steps: ', self.steps)
        while step < self.steps:
            start_time = time.time()
            x = next(self.train_iter)[0]
            
            batch_loss, pos_en, neg_en = self._run_step(x)

            # self._anneal_lr(step)

            curr_count += len(x)
            curr_loss += batch_loss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                loss = np.around(curr_loss / curr_count, 4)
                if np.isnan(loss):
                    print('Finding Nan')
                    break
                
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} Loss: {loss}, PosEn: {pos_en}, PosEn: {neg_en}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, loss]

                np.set_printoptions(suppress=True)
          
                curr_count = 0
                curr_loss = 0.0

                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'model.pt'))
  
                if (step + 1) % 10000 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_path, f'model_{step+1}.pt'))

            with torch.no_grad():
                for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
                    ema_p.mul_(0.999).add_(p, alpha=0.001)

            step += 1
            # end_time = time.time()
            # print('Time: ', end_time - start_time)

def train(
    model_save_path,
    real_data_path,
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    task_type = 'binclass',
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    emb_dim = 4,
    device = torch.device('cuda:0'),
    seed = 0,
    change_val = False,
):
    real_data_path = os.path.normpath(real_data_path)
    
    T = src.Transformations(**T_dict)
    
    dataset = make_dataset(
        real_data_path,
        T,
        task_type = task_type,
        change_val = False,
    )
    K = np.array(dataset.get_category_sizes('train'))
    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    print(K)
    print(num_numerical_features)

    # X_num = dataset.X_num['train']
    # print(np.mean(X_num), np.min(X_num), np.max(X_num))
    # exit()

    train_loader = src.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    in_dim = num_numerical_features + len(K) * emb_dim
    net = MLPScore(in_dim, [1024] * 3 + [1]).to(device)
    model = EBM(net, emb_dim, num_numerical_features, K).to(device)

    device1 = device
    class Args:
        nume_size = num_numerical_features
        num_classes = K
        device = device1
    args = Args()

    trainer = Trainer(
        model,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        model_save_path=model_save_path,
        device=device,
        args = args
    )
    trainer.run_loop()

    torch.save(trainer.ema_model.state_dict(), os.path.join(model_save_path, 'model_ema.pt'))
    trainer.loss_history.to_csv(os.path.join(model_save_path, 'loss.csv'), index=False)


