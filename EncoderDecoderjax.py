#attention mechanism written in jax and stax libraries for 2D data

import jax.numpy as jnp
from jax import random
from jax import jit
from jax import grad
from jax import vmap
from jax import lax
from jax.experimental import stax
from functools import partial

def DotProductAttention(query, key, value, mask):
  """Dot product self-attention.
  Args:
    query: array of representations
    key: array of representations
    value: array of representations
    mask: attention-mask, gates attention
  """
  depth = query.shape[-1]
  dots = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / jnp.sqrt(depth)
  if mask is not None:
    dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))
  attention = jnp.matmul(jnp.softmax(dots, axis=-1), value)
  return attention

def MultiHeadDotProductAttention(query, key, value, mask, num_heads=8):
  """Dot product self-attention.
  Args:
    query: array of representations
    key: array of representations
    value: array of representations
    mask: attention-mask, gates attention
    num_heads: number of attention heads
  """
  depth = query.shape[-1]
  query = jnp.reshape(query, (-1, query.shape[-2], num_heads, depth // num_heads))
  key = jnp.reshape(key, (-1, key.shape[-2], num_heads, depth // num_heads))
  value = jnp.reshape(value, (-1, value.shape[-2], num_heads, depth // num_heads))
  dots = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / jnp.sqrt(depth // num_heads)
  if mask is not None:
    dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))
  attention = jnp.matmul(jnp.softmax(dots, axis=-1), value)
  attention = jnp.reshape(attention, (-1, query.shape[1], depth))
  return attention

def ResidualConnection(x, y, dropout, mode):
  """Residual connection.
  Args:
    x: input
    y: output
    dropout: dropout rate
    mode: 'train' or 'eval'
  """
  if x.shape[-1] == y.shape[-1]:
    return jnp.dropout(x + y, rate=dropout) if mode == 'train' else x + y
  else:
    return jnp.dropout(x, rate=dropout) + y if mode == 'train' else x + y

def FeedForward(x, intermediate_size, hidden_size, dropout, mode):
  """Feed-forward block with layer normalization at start.
  Args:
    x: input
    intermediate_size: dimension of the intermediate layer
    hidden_size: dimension of the hidden layer
    dropout: dropout rate
    mode: 'train' or 'eval'
  """
  intermediate_activation = jnp.dot(x, stax.Dense(intermediate_size))
  hidden_activation = jnp.dot(intermediate_activation, stax.Dense(hidden_size))
  return ResidualConnection(
      x=layer_norm(x=x,
                   center=True,
                   scale=True,
                   activation_fn=hidden_activation),
      y=hidden_activation,
      dropout=dropout,
      mode=mode)

def layer_norm(x, center=True, scale=True, activation_fn=None):
  """Layer normalizes a 2D tensor along its second axis, which corresponds to
  normalizing within a layer.
  Args:
    x: Tensor with 2 dimensions.
    center: bool Tensor to indicate whether to offset by the mean.
    scale: bool Tensor to indicate whether to scale by the variance.
    activation_fn: Optional activation function.
  Returns:
    A normalized Tensor with the same shape as x.
  """
  if center:
    beta = jnp.zeros(shape=x.shape[-1:])
  else:
    beta = None
  if scale:
    gamma = jnp.ones(shape=x.shape[-1:])
  else:
    gamma = None
  mean = jnp.mean(x, axis=-1, keepdims=True)
  variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
  norm_x = (x - mean) * jnp.rsqrt(variance + jnp.finfo(jnp.float32).eps)
  return activation_fn(norm_x * gamma + beta) if activation_fn else norm_x

def EncoderBlock(x,
                 intermediate_size,
                 hidden_size,
                 num_heads,
                 dropout,
                 mode,
                 ff_activation=jnp.nn.relu):
  """A network with a feed-forward architecture for encoding sequences.
  Args:
    x: input
    intermediate_size: dimension of the intermediate layer
    hidden_size: dimension of the hidden layer
    num_heads: number of heads for multi-head attention
    dropout: dropout rate
    mode: 'train' or 'eval'
    ff_activation: activation function for feed-forward layer
  """
  x = layer_norm(x, center=False, scale=False, activation_fn=x)
  x = MultiHeadDotProductAttention(query=x,
                                   key=x,
                                   value=x,
                                   mask=None,
                                   num_heads=num_heads)
  x = layer_norm(x, center=False, scale=False, activation_fn=x)
  x = FeedForward(x=x,
                  intermediate_size=intermediate_size,
                  hidden_size=hidden_size,
                  dropout=dropout,
                  mode=mode,
                  ff_activation=ff_activation)
  return x

def DecoderBlock(x,
                 encoded_x,
                 intermediate_size,
                 hidden_size,
                 num_heads,
                 dropout,
                 mode,
                 ff_activation=jnp.nn.relu):
  """A network with a feed-forward architecture for decoding sequences.
  Args:
    x: input
    encoded_x: output from the encoder
    intermediate_size: dimension of the intermediate layer
    hidden_size: dimension of the hidden layer
    num_heads: number of heads for multi-head attention
    dropout: dropout rate
    mode: 'train' or 'eval'
    ff_activation: activation function for feed-forward layer
  """
  x = layer_norm(x, center=False, scale=False, activation_fn=x)
  x = MultiHeadDotProductAttention(query=x,
                                   key=x,
                                   value=x,
                                   mask=None,
                                   num_heads=num_heads)
  x = layer_norm(x, center=False, scale=False, activation_fn=x)
  x = MultiHeadDotProductAttention(query=x,
                                   key=encoded_x,
                                   value=encoded_x,
                                   mask=None,
                                   num_heads=num_heads)
  x = layer_norm(x, center=False, scale=False, activation_fn=x)
  x = FeedForward(x=x,
                  intermediate_size=intermediate_size,
                  hidden_size=hidden_size,
                  dropout=dropout,
                  mode=mode,
                  ff_activation=ff_activation)
  return x

def TransformerEncoder(x, num_layers, intermediate_size, hidden_size, num_heads,
                       dropout, mode, ff_activation=jnp.nn.relu):
  """A stack of transformer encoder blocks.
  Args:
    x: input
    num_layers: number of encoder blocks
    intermediate_size: dimension of the intermediate layer
    hidden_size: dimension of the hidden layer
    num_heads: number of heads for multi-head attention
    dropout: dropout rate
    mode: 'train' or 'eval'
    ff_activation: activation function for feed-forward layer
  """
  for _ in range(num_layers):
    x = EncoderBlock(x=x,
                     intermediate_size=intermediate_size,
                     hidden_size=hidden_size,
                     num_heads=num_heads,
                     dropout=dropout,
                     mode=mode,
                     ff_activation=ff_activation)
  return x

def TransformerDecoder(x,
                       encoded_x,
                       num_layers,
                       intermediate_size,
                       hidden_size,
                       num_heads,
                       dropout,
                       mode,
                       ff_activation=jnp.nn.relu):
  """A stack of transformer decoder blocks.
  Args:
    x: input
    encoded_x: output from the encoder
    num_layers: number of decoder blocks
    intermediate_size: dimension of the intermediate layer
    hidden_size: dimension of the hidden layer
    num_heads: number of heads for multi-head attention
    dropout: dropout rate
    mode: 'train' or 'eval'
    ff_activation: activation function for feed-forward layer
  """
  for _ in range(num_layers):
    x = DecoderBlock(x=x,
                     encoded_x=encoded_x,
                     intermediate_size=intermediate_size,
                     hidden_size=hidden_size,
                     num_heads=num_heads,
                     dropout=dropout,
                     mode=mode,
                     ff_activation=ff_activation)
  return x

def Transformer(vocab_size,
                num_layers=6,
                intermediate_size=512,
                hidden_size=512,
                num_heads=8,
                dropout=0.1,
                max_len=2048,
                mode='train',
                ff_activation=jnp.nn.relu):
  """Transformer language model.
  Args:
    vocab_size: size of the vocabulary
    num_layers: number of decoder blocks
    intermediate_size: dimension of the intermediate layer
    hidden_size: dimension of the hidden layer
    num_heads: number of heads for multi-head attention
    dropout: dropout rate
    max_len: maximal length
    mode: 'train' or 'eval'
    ff_activation: activation function for feed-forward layer
  """
  embedding_initializer = 'uniform'
  embedding_shape = (vocab_size, hidden_size)
  input_shape = (-1, max_len)
  embedded_input = stax.embedding(
      ids=random.randint(key=random.PRNGKey(0),
                         minval=0,
                         maxval=vocab_size - 1,
                         shape=input_shape),
      vocab_size=vocab_size,
      features=hidden_size,
      dimension=hidden_size,
      embedding_init=embedding_initializer,
      dtype=jnp.float32,
      name='embedding')
  x = TransformerEncoder(x=embedded_input,
                         num_layers=num_layers,
                         intermediate_size=intermediate_size,
                         hidden_size=hidden_size,
                         num_heads=num_heads,
                         dropout=dropout,
                         mode=mode,
                         ff_activation=ff_activation)
  x = layer_norm(x, center=False, scale=False, activation_fn=x)
  logits = stax.Dense(hidden_size,
                      W_init=embedding_initializer,
                      b_init=jnp.zeros,
                      dtype=jnp.float32)(x)
  return logits

def TransformerLM(vocab_size,
                  num_layers=6,
                  intermediate_size=512,
                  hidden_size=512,
                  num_heads=8,
                  dropout=0.1,
                  max_len=2048,
                  mode='train',
                  ff_activation=jnp.nn.relu):
  """Transformer language model.
  Args:
    vocab_size: size of the vocabulary
    num_layers: number of decoder blocks
    intermediate_size: dimension of the intermediate layer
    hidden_size: dimension of the hidden layer
    num_heads: number of heads for multi-head attention
    dropout: dropout rate
    max_len: maximal length
    mode: 'train' or 'eval'
    ff_activation: activation function for feed-forward layer
  """
  logits = Transformer(vocab_size,
                       num_layers,
                       intermediate_size,
                       hidden_size,
                       num_heads,
                       dropout,
                       max_len,
                       mode,
                       ff_activation)
  return stax.logsoftmax(logits)

def bottleneck(x, bottleneck_bits):
    """Creates a bottleneck layer for the given activations."""
    with nn.stochastic(jax.random.PRNGKey(0)):
      return nn.tanh(
          nn.Dense(x, features=bottleneck_bits, dtype=jnp.float32))

  def logistic_compression(x, bottleneck_bits):
    """Creates a logistic compression layer for the given activations."""
    with nn.stochastic(jax.random.PRNGKey(0)):
      return nn.Dense(x, features=bottleneck_bits, dtype=jnp.float32)

  def compression_function(x, bottleneck_bits):
    """Compression function for a general shape."""
    x_shape = x.shape
    x = nn.reshape(x, (-1, x_shape[-1]))
    if bottleneck_bits == 0:
      x = nn.reshape(x, x_shape)
      return x
    x = bottleneck(x, bottleneck_bits)
    x = nn.reshape(x, x_shape)
    return x

  def compression_function_per_bit(x, compression_spec):
    """Compression function for a scalar compression spec."""
    return jnp.where(
        x > 0,
        logistic_compression(x, compression_spec.bits_per_feature),
        nn.Dense(x, features=compression_spec.bits_per_feature,
                 dtype=jnp.float32))
