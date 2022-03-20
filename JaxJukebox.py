#implementing the jukebox music model using jax model.
#we will tackel the problem in the following steps:
#import the required libraries

import jax.numpy as np
from jax import random
from jax import grad, jit, vmap
from jax import lax
from jax import ops
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from jax.nn import softmax
from jax.experimental import stax
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax, Dropout
from jax.nn import softmax
from jax.nn.initializers import zeros
from jax.experimental import optimizers
from jax.tree_util import tree_multimap  # Element-wise manipulation of collections of numpy arrays

#create the mask. returns a mask of shape 1 x 1 x q_l x kv_l or None if masking is not needed
def create_mask(q_l, kv_l):
  if q_l > kv_l:
    return None
  mask = np.ones((1, 1, q_l, kv_l))
  for i in range(q_l):
    mask[:, :, i, i] = 0
  return mask

class FactoredAttention(object):
  def __init__(self, q_l, kv_l, heads, dropout=0.1):
    self.q_l = q_l
    self.kv_l = kv_l
    self.heads = heads
    self.mask = create_mask(q_l, kv_l)
    self.dropout = dropout
    self.attention_fn = stax.serial(
        Dense(512), Relu, Dense(512), Relu, Dense(512), Relu, Dense(512),
        Dropout(dropout), Dense(kv_l * heads, None))
    self.out_fn = stax.serial(Dense(512), Relu, Dense(512), Relu, Dense(512),
                              Relu, Dense(512), Relu, Dense(512), Relu,
                              Dense(q_l))

  def attention(self, q, k, v, mask=None):
    # q, k, v have shape (bs, q_l, kv_l)
    # mask has shape (bs, 1, 1, q_l, kv_l)
    # output has shape (bs, heads, q_l, kv_l)
    # Note: this only works for self attention!
    # Note: this does not do masked attention.
    # Note: this does not do dropout.
    bs = q.shape[0]
    heads = self.heads
    kv_l = self.kv_l
    q_l = self.q_l
    assert q.shape == (bs, q_l, kv_l)
    assert k.shape == (bs, q_l, kv_l)
    assert v.shape == (bs, q_l, kv_l)
    assert mask is None or mask.shape == (bs, 1, 1, q_l, kv_l)
    # Compute attention weights for each head.
    # w has shape (bs, heads, q_l, kv_l)
    w = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(kv_l)
    if mask is not None:
      w = w * mask + -1e9 * (1 - mask)
    w = softmax(w)
    # Compute attention output.
    # o has shape (bs, heads, q_l, kv_l)
    o = np.matmul(w, v)
    # Concatenate heads.
    # o has shape (bs, q_l, kv_l * heads)
    o = np.reshape(o, (bs, q_l, kv_l * heads))
    # Run output projection.
    # output has shape (bs, q_l, kv_l)
    output = self.out_fn(o)
    return output

  def init_attention(self, key, value, query):
    # key, value, query have shape (bs, kv_l, heads)
    bs = key.shape[0]
    kv_l = self.kv_l
    heads = self.heads
    assert key.shape == (bs, kv_l, heads)
    assert value.shape == (bs, kv_l, heads)
    assert query.shape == (bs, kv_l, heads)
    # Compute attention weights for each head.
    # w has shape (bs, heads, kv_l, kv_l)
    w = np.matmul(query, np.transpose(key, (0, 2, 1))) / np.sqrt(kv_l)
    if self.mask is not None:
      w = w * self.mask + -1e9 * (1 - self.mask)
    w = softmax(w)
    # Compute attention output.
    # o has shape (bs, heads, kv_l, kv_l)
    o = np.matmul(w, value)
    # Concatenate heads.
    # o has shape (bs, kv_l, heads * kv_l)
    o = np.reshape(o, (bs, kv_l, heads * kv_l))
    # Run output projection.
    # output has shape (bs, kv_l, kv_l)
    output = self.attention_fn(o)
    return output

  def forward(self, key, value, query):
    # key, value, query have shape (bs, kv_l, heads)
    bs = key.shape[0]
    kv_l = self.kv_l
    heads = self.heads
    assert key.shape == (bs, kv_l, heads)
    assert value.shape == (bs, kv_l, heads)
    assert query.shape == (bs, kv_l, heads)
    # Compute attention weights for each head.
    # w has shape (bs, heads, kv_l, kv_l)
    w = np.matmul(query, np.transpose(key, (0, 2, 1))) / np.sqrt(kv_l)
    if self.mask is not None:
      w = w * self.mask + -1e9 * (1 - self.mask)
    w = softmax(w)
    # Compute attention output.
    # o has shape (bs, heads, kv_l, kv_l)
    o = np.matmul(w, value)
    # Concatenate heads.
    # o has shape (bs, kv_l, heads * kv_l)
    o = np.reshape(o, (bs, kv_l, heads * kv_l))
    # Run output projection.
    # output has shape (bs, kv_l, kv_l)
    output = self.attention_fn(o)
    # Run projection.
    # output has shape (bs, kv_l, heads)
    output = np.transpose(output, (0, 2, 1))
    output = np.reshape(output, (bs, kv_l, heads))
    # Run softmax.
    # output has shape (bs, kv_l, heads)
    output = softmax(output)
    # Run final output.
    # output has shape (bs, heads, kv_l)
    output = np.matmul(output, value)
    # Concatenate heads.
    # output has shape (bs, kv_l, heads)
    output = np.reshape(output, (bs, kv_l, heads))
    # Run output projection.
    # output has shape (bs, kv_l, kv_l)
    output = self.out_fn(output)
    return output


def main():
  # Create model.
  model = FactoredAttention(q_l=8, kv_l=8, heads=2)
  # Create data.
  key = random.normal(random.PRNGKey(0), (1, 8, 2))
  value = random.normal(random.PRNGKey(0), (1, 8, 2))
  query = random.normal(random.PRNGKey(0), (1, 8, 2))
  # Compute forward pass.
  output = model.forward(key, value, query)
  print(output)
  print(output.shape)
  # Compute backward pass.
  loss = lambda x: np.sum(x**2)
  grad_fn = grad(loss)
  grads = grad_fn(output)
  print(grads)
  print(grads.shape)


if __name__ == '__main__':
  main()
