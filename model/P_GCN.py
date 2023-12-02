import math

import torch
import torch.nn as nn
import numpy as np

from .backbone.resnet import resnet101


def gen_A(num_classes, t, labels):
    _adj = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.float)
    _nums = np.zeros((labels.shape[1],), dtype=np.float)

    for index in range(labels.shape[0]):
        indexs = np.where(labels[index] == 1)[0]
        for i in indexs:
            _nums[i] += 1
            for j in indexs:
                _adj[i, j] += 1

    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    #_adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    #_adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.bmm(input, self.weight.unsqueeze(0).repeat(input.size(0), 1, 1))
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class PGCN(nn.Module):
    def __init__(self, num_classes, out_channel=2048, adj=None):
        super(PGCN, self).__init__()

        self.backbone = resnet101(pretrained=True, avg_pool=False)

        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)
        self.out_channel = out_channel
        if out_channel != 2048:
            self.transition = nn.Sequential(nn.Conv2d(2048, out_channel, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(out_channel),)

        self.fc = nn.Linear(out_channel, num_classes)
        self.gc1 = GraphConvolution(out_channel, out_channel)
        self.gc2 = GraphConvolution(out_channel, 1)
        self.relu = nn.LeakyReLU(0.2)

        self.A = nn.Parameter(torch.from_numpy(adj).float(), requires_grad=False)

    def forward(self, input):
        batch_size = input.size(0)
        feature = self.backbone(input)
        feature = self.pooling(feature)
        if self.out_channel != 2048:
            feature = self.transition(feature)
        
        y = self.fc(feature.view(batch_size, -1))

        x = feature.view(batch_size, 1, -1).repeat((1, self.num_classes, 1))
        x = x.contiguous()

        adj = gen_adj(self.A).detach()
        adj = adj.unsqueeze(0).repeat((batch_size, 1, 1))

        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = x.view(batch_size, -1)

        out = x + y
        return out
