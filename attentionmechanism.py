import torch
import torch.nn as nn
import torch.nn.functional as F

# assume we have some tensor x with size (b, t, k) -> (minibatch dimension, number of vectors, dimension)
x = ...


raw_weights = torch.bmm(x, x.transpose(1, 2))

weights = F.Softmax(raw_weights, dim=2)

y = torch.bmm(weights, x)