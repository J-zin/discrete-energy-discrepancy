import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity='elu', act_last=None, bn=False, dropout=-1):
        super(MLP, self).__init__()
        self.act_last = act_last
        self.nonlinearity = nonlinearity
        self.input_dim = input_dim
        self.bn = bn

        if isinstance(hidden_dims, str):
            hidden_dims = list(map(int, hidden_dims.split("-")))
        assert len(hidden_dims)
        hidden_dims = [input_dim] + hidden_dims
        self.output_size = hidden_dims[-1]

        list_layers = []

        for i in range(1, len(hidden_dims)):
            list_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if i + 1 < len(hidden_dims):  # not the last layer
                if self.bn:
                    bnorm_layer = nn.BatchNorm1d(hidden_dims[i])
                    list_layers.append(bnorm_layer)
                list_layers.append(NONLINEARITIES[self.nonlinearity])
                if dropout > 0:
                    list_layers.append(nn.Dropout(dropout))
            else:
                if act_last is not None:
                    list_layers.append(NONLINEARITIES[act_last])

        self.main = nn.Sequential(*list_layers)

    def forward(self, z):
        x = self.main(z)
        return x

class MLPScore(nn.Module):
    def __init__(self, input_dim, hidden_dims, scale=1.0, nonlinearity='swish', act_last=None, bn=False, dropout=-1, bound=-1):
        super(MLPScore, self).__init__()
        self.scale = scale
        self.bound = bound
        self.mlp = MLP(input_dim, hidden_dims, nonlinearity, act_last, bn, dropout)

    def forward(self, z):
        raw_score = self.mlp(z.float() / self.scale)
        if self.bound > 0:
            raw_score = torch.clamp(raw_score, min=-self.bound, max=self.bound)
        return raw_score

class EBM(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.net = net
        self.emb_dim = args.emb_dim
        self.num_size = args.nume_size
        self.cat_size = len(args.num_classes)
        self.categories = args.num_classes

        self.emb_list = nn.ModuleList([nn.Linear(num_class, self.emb_dim) for num_class in self.categories])

    def encode_input(self, x):
        x_num = x[:, :self.num_size]
        x_cat = x[:, self.num_size:]
        if x_cat.shape[1] == self.cat_size:
            """ if x_cat is integer """
            x_cat = [
                emb(F.one_hot(x_cat[:, i].long(), num_classes=self.categories[i]).float()) 
                for i, emb in enumerate(self.emb_list)]
            x_cat = torch.cat(x_cat, dim=1)
        else:
            """ if x_cat is one-hot encoding """
            # for debugging
            # x_cat  = [F.one_hot(x_cat[:, i].long(), num_classes=self.categories[i]).float() for i in range(self.cat_size)]
            # x_cat = torch.cat(x_cat, dim=1)
            x_cat_embs = []
            for i, emb in enumerate(self.emb_list):
                ind_begin, ind_end = sum(self.categories[:i]), sum(self.categories[:i+1])
                x_cat_onehot = x_cat[:, ind_begin:ind_end]
                x_cat_embs.append(emb(x_cat_onehot))
            x_cat = torch.cat(x_cat_embs, dim=1)
            
        x = torch.cat([x_num, x_cat], dim=1)
        return x

    def forward(self, x):
        '''we define p(x) = exp(-f(x)) / Z, the output of net is f(x)
        x: [bs, dim]
        '''
        emb = self.encode_input(x)
        logp = self.net(emb).squeeze()
        
        return logp
