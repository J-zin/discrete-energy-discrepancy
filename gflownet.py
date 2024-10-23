import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def make_mlp(l, act=nn.LeakyReLU(), tail=[], with_bn=False):
    """makes an MLP with no top layer activation"""
    net = nn.Sequential(*(sum(
        [[nn.Linear(i, o)] + (([nn.BatchNorm1d(o), act] if with_bn else [act]) if n < len(l) - 2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []
    ) + tail))
    return net


def mlp_ebm(nin, nint=256, nout=1):
    return nn.Sequential(
        nn.Linear(nin, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nout),
    )

def get_GFlowNet(type, xdim, args, device, net=None):
    if type == "tbrf":
        # return GFlowNet_Randf_TB(xdim=xdim, args=args, device=device, net=net)
        raise NotImplementedError
    elif type == "tblb":
        return GFlowNet_LearnedPb_TB(xdim=xdim, args=args, device=device, net=net)
    else:
        raise NotImplementedError

class GFlowNet_LearnedPb_TB:
    def __init__(self, xdim, args, device, net=None):
        self.xdim = xdim
        self._hops = 0.
        # (bs, data_dim) -> (bs, data_dim)
        if net is None:
            self.model = make_mlp([xdim] + [args.hid] * args.hid_layers +
                              [3 * xdim], act=(nn.LeakyReLU() if args.leaky else nn.ReLU()), with_bn=args.gfn_bn)
        else:
            self.model = net
        self.model.to(device)

        self.logZ = nn.Parameter(torch.tensor(0.))
        self.logZ.to(device)
        self.device = device

        self.exp_temp = args.temp
        self.rand_coef = args.rand_coef  # involving exploration
        self.init_zero = args.init_zero
        self.clip = args.clip
        self.l1loss = args.l1loss

        self.replay = None
        self.tau = args.tau if hasattr(args, "tau") else -1

        self.train_steps = args.train_steps
        param_list = [{'params': self.model.parameters(), 'lr': args.glr},
                      {'params': self.logZ, 'lr': args.zlr}]
        if args.opt == "adam":
            self.optimizer = torch.optim.Adam(param_list)
        elif args.opt == "sgd":
            self.optimizer = torch.optim.SGD(param_list, momentum=args.momentum)
        
        self.print_every = args.print_every

    def sample(self, batch_size):
        self.model.eval()
        if self.init_zero:
            x = torch.zeros((batch_size, self.xdim)).to(self.device)
        else:
            x = -1 * torch.ones((batch_size, self.xdim)).to(self.device)

        for step in range(self.xdim + 1):
            logits = self.model(x)
            add_logits, del_logits = logits[:, :2 * self.xdim], logits[:, 2 * self.xdim:]

            # those have been edited
            if self.init_zero:
                mask = (x != 0).unsqueeze(2).repeat(1, 1, 2).reshape(batch_size, 2 * self.xdim).float()
            else:
                mask = (x > -0.5).unsqueeze(2).repeat(1, 1, 2).reshape(batch_size, 2 * self.xdim).float()
            add_prob = (add_logits - 1e9 * mask).float().softmax(1)

            if step < self.xdim:
                # add_prob = add_prob ** (1 / self.exp_temp)
                add_sample = add_prob.multinomial(1)  # row sum not need to be 1
                if self.init_zero:
                    add_locs, add_values = add_sample // 2, 2 * (add_sample % 2) - 1
                else:
                    add_locs, add_values = add_sample // 2, add_sample % 2

                x = x.scatter(1, add_locs, add_values.float())
        return x

    def cal_logp(self, data, num: int):
        logp_ls = []
        for _ in range(num):
            _, _, _, mle_loss, data_log_pb = tb_mle_learnedpb_loss(lambda inp: torch.tensor(0.).to(self.device), self, data.shape[0], back_ratio=1, data=data)
            logpj = - mle_loss.detach().cpu() - data_log_pb.detach().cpu()
            logp_ls.append(logpj.reshape(logpj.shape[0], -1))
        batch_logp = torch.logsumexp(torch.cat(logp_ls, dim=1), dim=1)  # (bs,)

        return batch_logp.mean() - torch.tensor(num).log()

    def evaluate(self, loader, preprocess, num, use_tqdm=False):
        logps = []
        if use_tqdm:
            pbar = tqdm(loader)
        else:
            pbar = loader

        if hasattr(pbar, "set_description"):
            pbar.set_description("Calculating likelihood")
        self.model.eval()
        for x, _ in pbar:
            x = preprocess(x.to(self.device))
            logp = self.cal_logp(x, num)
            logps.append(logp.reshape(-1))
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"logp": f"{torch.cat(logps).mean().item():.2f}"})

        return torch.cat(logps).mean()

    def train(self, batch_size, scorer, itr=0, data=None, back_ratio=0.):
        curr_lr = self.optimizer.param_groups[0]['lr']

        train_loss = []
        train_mle_loss = []
        train_logZ = []
        self.model.train()
        self.model.zero_grad()
        torch.cuda.empty_cache()

        gfn_loss, forth_loss, back_loss, mle_loss, data_log_pb = \
            tb_mle_learnedpb_loss(scorer, self, batch_size, back_ratio=back_ratio, data=data)
        gfn_loss, forth_loss, back_loss, mle_loss, data_log_pb = \
            gfn_loss.mean(), forth_loss.mean(), back_loss.mean(), mle_loss.mean(), data_log_pb.mean()
        
        loss = gfn_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip, norm_type="inf")
        self.optimizer.step()

        train_loss.append(gfn_loss.item())
        train_mle_loss.append(mle_loss.item())
        train_logZ.append(self.logZ.item())

        if itr % self.print_every == 0:
            print({
                "Itr": itr,
                "LR": "{:.1e}".format(curr_lr),
                "MLE": "{:.2e}".format(mle_loss.item()),
                "GFN": "{:.2e}".format(gfn_loss.item()),
                "Forth": "{:.2e}".format(forth_loss.item()),
                "Back": "{:.2e}".format(back_loss.item()),
                "LogZ": "{:.2e}".format(self.logZ.item()),
            })
            
        return np.mean(train_loss), np.mean(train_logZ)


def tb_mle_learnedpb_loss(ebm_model, gfn, batch_size, back_ratio=0., data=None):
    # traj is from s0 -> sf, sample by current gfn policy
    if back_ratio < 1.:
        if gfn.init_zero:
            x = torch.zeros((batch_size, gfn.xdim)).to(gfn.device)
        else:
            # -1 denotes "have not been edited"
            x = -1 * torch.ones((batch_size, gfn.xdim)).to(gfn.device)
        
        # forth_loss = 0.
        log_pb = 0.
        log_pf = 0.
        for step in range(gfn.xdim + 1):
            logits = gfn.model(x)
            add_logits, del_logits = logits[:, :2 * gfn.xdim], logits[:, 2 * gfn.xdim:]
            
            if step > 0:
                if gfn.init_zero:
                    mask = (x.abs() < 1e-8).float()
                else:
                    mask = (x < 0).float()
                log_pb = log_pb + (del_logits - 1e9 * mask).log_softmax(1).gather(1, add_locs).squeeze(1)

            if step < gfn.xdim:
                # mask those that have been edited
                if gfn.init_zero:
                    mask = (x != 0).unsqueeze(2).repeat(1, 1, 2).reshape(batch_size, 2 * gfn.xdim).float()
                else:
                    mask = (x > -0.5).unsqueeze(2).repeat(1, 1, 2).reshape(batch_size, 2 * gfn.xdim).float()

                add_logits = (add_logits - 1e9 * mask).float()
                add_prob = add_logits.softmax(1)

                add_prob = add_prob ** (1 / gfn.exp_temp)
                add_prob = add_prob / (1e-9 + add_prob.sum(1, keepdim=True))
                add_prob = (1 - gfn.rand_coef) * add_prob + \
                           gfn.rand_coef * (1 - mask) / (1e-9 + (1 - mask).sum(1)).unsqueeze(1)

                add_sample = add_prob.multinomial(1)
                if gfn.init_zero:
                    add_locs, add_values = add_sample // 2, 2 * (add_sample % 2) - 1
                else:
                    add_locs, add_values = add_sample // 2, add_sample % 2
                # P_F
                log_pf = log_pf + add_logits.log_softmax(1).gather(1, add_sample).squeeze(1)
                # update x
                x = x.scatter(1, add_locs, add_values.float())
   
        assert torch.all(x != 0) if gfn.init_zero else torch.all(x >= 0)

        score_value = ebm_model(x)
        if gfn.l1loss:
            forth_loss = F.smooth_l1_loss(gfn.logZ + log_pf - log_pb - score_value, torch.zeros_like(score_value))
        else:
            forth_loss = ((gfn.logZ + log_pf - log_pb - score_value) ** 2)
    else:
        forth_loss = torch.tensor(0.).to(gfn.device)

    mle_loss = torch.tensor(0.).to(gfn.device)  # log_pf
    if back_ratio <= 0.:
        data_log_pb = torch.tensor(0.).to(gfn.device)
        back_loss = torch.tensor(0.).to(gfn.device)
    else:
        assert data is not None
        x = data
        batch_size = x.size(0)
        data_log_pb = torch.zeros(batch_size).to(gfn.device)

        for step in range(gfn.xdim + 1):
            logits = gfn.model(x)
            add_logits, del_logits = logits[:, :2 * gfn.xdim], logits[:, 2 * gfn.xdim:]

            if step > 0:
                if gfn.init_zero:
                    mask = (x != 0).unsqueeze(2).repeat(1, 1, 2).reshape(batch_size, 2 * gfn.xdim).float()
                else:
                    mask = (x > -0.5).unsqueeze(2).repeat(1, 1, 2).reshape(batch_size, 2 * gfn.xdim).float()

                add_sample = del_locs * 2 + (deleted_values == 1).long()  # whether it's init_zero, this holds true
                add_logits = (add_logits - 1e9 * mask).float()
                mle_loss = mle_loss + add_logits.log_softmax(1).gather(1, add_sample).squeeze(1)

            if step < gfn.xdim:
                if gfn.init_zero:
                    # mask = (x == 0).float()
                    mask = (x.abs() < 1e-8).float()
                else:
                    mask = (x < -0.5).float()
                del_logits = (del_logits - 1e9 * mask).float()
                del_prob = del_logits.softmax(1)
                del_prob = (1 - gfn.rand_coef) * del_prob + gfn.rand_coef * (1 - mask) / (1e-9 + (1 - mask).sum(1)).unsqueeze(1)
                del_locs = del_prob.multinomial(1)  # row sum not need to be 1
                deleted_values = x.gather(1, del_locs)
                data_log_pb = data_log_pb + del_logits.log_softmax(1).gather(1, del_locs).squeeze(1)

                del_values = torch.ones(batch_size, 1).to(gfn.device) * (0 if gfn.init_zero else -1)
                x = x.scatter(1, del_locs, del_values)

        if gfn.l1loss:
            back_loss = F.smooth_l1_loss(gfn.logZ + mle_loss - data_log_pb - ebm_model(data).detach(), torch.zeros_like(mle_loss))
        else:
            back_loss = ((gfn.logZ + mle_loss - data_log_pb - ebm_model(data).detach()) ** 2)

    gfn_loss = (1 - back_ratio) * forth_loss + back_ratio * back_loss
    mle_loss = - mle_loss

    return gfn_loss, forth_loss, back_loss, mle_loss, data_log_pb
