import torch
import torch.nn as nn
import torch.nn.functional as F

#k stands for dimension of vectors t
class selfattention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads

        # These compute the queries, keys and values for all heads (as a single concatenated vector)
        self.tokeys = nn.Linear(k, heads * k, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)

        # This unifies the outputs of the different heads into
        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toqueries(x).view(b, t, k, h)
        keys = self.tokeys(x).view(b, t, k, h)
        values = self.tovalues(x).view(b, t, k, h)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h. t. k)
        values = values.transpose(1, 2).contiguous().view(b * h. t. k)

        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values):view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, h, t, k)
        return self.unifyheads(out)