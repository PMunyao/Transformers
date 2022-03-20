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

# Import FusedLayerNorm if we have apex, otherwise use regular LayerNorm
try:
    from apex.normalization import FusedLayerNorm
    print("Using apex FusedLayerNorm")
except ImportError:
    from jax.experimental import stax
    FusedLayerNorm = stax.LayerNorm
    print("Using jax regular LayerNorm")

class LayerNorm(FusedLayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def __call__(self, x):
        if self.elementwise_affine:
            scale, bias = self.params
            return (x - self.normalize(x)) * scale + bias
        else:
            return x - self.normalize(x)

    def normalize(self, x):
        if x.dtype not in (np.float32, np.float64):
            x = x.astype(np.float32)
        mean, var = np.mean(x, axis=-1, keepdims=True), np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps)
        
    def forward(self, input):
        if self.elementwise_affine:
            scale, bias = self.params
            return (input - self.normalize(input)) * scale + bias
        else:
            return input - self.normalize(input)
            
class Conv1D(LayerNorm):
    def __init__(self, num_channels, kernel_size, strides=1, padding='SAME',
                 dilation=1, w_init_gain='linear', use_bias=True):
        super(Conv1D, self).__init__((1, num_channels, 1))
        self.padding = padding.upper()
        self.strides = strides
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_bias = use_bias
        self.w_init_gain = w_init_gain
        self.num_channels = num_channels

    def forward(self, x):
        if self.w_init_gain == 'linear':
            w_init = stax.linear
        elif self.w_init_gain == 'relu':
            w_init = stax.relu
        else:
            raise ValueError(
                'Unknown w_init_gain value "%s".'
                "Should be either 'linear' or 'relu'" % self.w_init_gain
            )

        return lax.conv_general_dilated(
            x,
            self.params[0],
            self.padding,
            strides=self.strides,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation,
            dimension_numbers=("NWC", "WIO", "NWC"),
            feature_group_count=1,
        )

class Mask(LayerNorm):
    def __init__(self, key_size, query_size, value_size):
        super(Mask, self).__init__((1, 1, value_size))
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size

    def forward(self, queries, keys, values):
        queries = queries[:, None, :]
        keys = keys[:, None, :]
        values = values[:, None, :]
        key_size = keys.shape[1]
        query_size = queries.shape[1]
        query_ones = np.ones((1, query_size, 1))
        key_ones = np.ones((1, key_size, 1))
        attention_mask = np.matmul(
            queries * query_ones * -1e9, key_ones.transpose([0, 2, 1])
        )
        return values * attention_mask


class MLP(LayerNorm):
    def __init__(self, hidden_size, output_size, dropout, w_init_gain="relu"):
        super(MLP, self).__init__((1, 1, hidden_size))
        self.linear_layer = Dense(
            hidden_size, output_size, w_init=stax.serial(stax.Dense(hidden_size), stax.Relu)
        )
        self.dropout = dropout
        self.w_init_gain = w_init_gain

    def forward(self, x):
        return self.linear_layer(x)

class ResAttnBlock(LayerNorm):
    def __init__(self, num_heads, dropout, hidden_size, key_size, value_size, output_size,
                 w_init_gain="relu"):
        super(ResAttnBlock, self).__init__((1, 1, hidden_size))
        self.dropout = dropout
        self.attn = Attention(num_heads, key_size, value_size, hidden_size, w_init_gain)
        self.linear_layer = Dense(
            hidden_size, output_size, w_init=stax.serial(stax.Dense(hidden_size), stax.Relu)
        )
        self.w_init_gain = w_init_gain

    def forward(self, x):
        x = self.attn(x)
        x = x[:, 0, :]
        x = self.linear_layer(x)
        return x

class Attention(LayerNorm):
    def __init__(self, num_heads, key_size, value_size, output_size, w_init_gain="relu"):
        super(Attention, self).__init__((1, 1, value_size))
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.w_init_gain = w_init_gain
        self.linear_layer_q = Dense(
            value_size,
            output_size,
            w_init=stax.serial(stax.Dense(value_size), stax.Relu),
        )
        self.linear_layer_k = Dense(
            value_size,
            output_size,
            w_init=stax.serial(stax.Dense(value_size), stax.Relu),
        )
        self.linear_layer_v = Dense(
            value_size,
            output_size,
            w_init=stax.serial(stax.Dense(value_size), stax.Relu),
        )

    def forward(self, x):
        queries = self.linear_layer_q(x)
        keys = self.linear_layer_k(x)
        values = self.linear_layer_v(x)
        queries = np.concatenate(np.split(queries, self.num_heads, axis=2), axis=0)
        keys = np.concatenate(np.split(keys, self.num_heads, axis=2), axis=0)
        values = np.concatenate(np.split(values, self.num_heads, axis=2), axis=0)
        attention_output = self._attention_forward(queries, keys, values)
        attention_output = np.concatenate(np.split(attention_output, self.num_heads, axis=0), axis=2)
        return attention_output

    def _attention_forward(self, queries, keys, values):
        """
        Forward pass of attention layer
        :param queries:
        :param keys:
        :param values:
        :return:
        """
        # Shape (batch_size, num_heads, seq_len, seq_len)
        scaled_similarities = np.matmul(queries, keys.swapaxes(1, 2)) / np.sqrt(self.key_size)
        # Shape (batch_size, num_heads, seq_len, seq_len)
        # Normalise the distributions, using the same mask for all heads.
        attention = softmax(scaled_similarities, axis=-1)
        attention = Dropout(self.dropout)(attention)
        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads dimension.
        # Shape (batch_size, num_heads, seq_len, value_size)
        outputs = np.matmul(attention, values)
        # Concatenate num_heads dimension to return shape (batch_size, seq_len, value_size * num_heads)
        outputs = np.concatenate(np.split(outputs, self.num_heads, axis=0), axis=-1)
        # Residual connection
        outputs += x
        # Normalise
        outputs = LayerNorm()(outputs)
        return outputs

class Transformer(
    LayerNorm
):
    def __init__(
        self,
        num_heads,
        key_size,
        value_size,
        hidden_size,
        output_size,
        num_layers,
        dropout,
        w_init_gain="relu",
    ):
        super(Transformer, self).__init__((1, 1, hidden_size))
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.w_init_gain = w_init_gain
        self.linear_layer_q = Dense(
            value_size,
            output_size,
            w_init=stax.serial(stax.Dense(value_size), stax.Relu),
        )
        self.linear_layer_k = Dense(
            value_size,
            output_size,
            w_init=stax.serial(stax.Dense(value_size), stax.Relu),
        )
        self.linear_layer_v = Dense(
            value_size,
            output_size,
            w_init=stax.serial(stax.Dense(value_size), stax.Relu),
        )

    def forward(self, x):
        queries = self.linear_layer_q(x)
        keys = self.linear_layer_k(x)
        values = self.linear_layer_v(x)
        queries = np.concatenate(np.split(queries, self.num_heads, axis=2), axis=0)
        keys = np.concatenate(np.split(keys, self.num_heads, axis=2), axis=0)
        values = np.concatenate(np.split(values, self.num_heads, axis=2), axis=0)
        attention_output = self._attention_forward(queries, keys, values)
        attention_output = np.concatenate(np.split(attention_output, self.num_heads, axis=0), axis=-1)
        return attention_output

    def _attention_forward(self, queries, keys, values):
        """
        Forward pass of attention layer
        :param queries:
        :param keys:
        :param values:
        :return:
        """
        # Shape (batch_size, num_heads, seq_len, seq_len)
        scaled_similarities = np.matmul(queries, keys.swapaxes(1, 2)) / np.sqrt(self.key_size)
        # Shape (batch_size, num_heads, seq_len, seq_len)
        # Normalise the distributions, using the same mask for all heads.
        attention = softmax(scaled_similarities, axis=-1)
        attention = Dropout(self.dropout)(attention)
        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads dimension.
        # Shape (batch_size, num_heads, seq_len, value_size)
        outputs = np.matmul(attention, values)
        # Concatenate num_heads dimension to return shape (batch_size, seq_len, value_size * num_heads)
        outputs = np.concatenate(np.split(outputs, self.num_heads, axis=0), axis=-1)
        # Residual connection
        outputs += x
        # Normalise
        outputs = LayerNorm()(outputs)
        return outputs
        
    def _attention_backward(
        self, grad_attn_output, queries, keys, values, attention_mask
    ):
        """
        Backward pass for attention layer
        :param grad_attn_output:
        :param queries:
        :param keys:
        :param values:
        :param attention_mask:
        :return:
        """
        # Reshape into (batch_size, num_heads, seq_len, value_size)
        outputs = np.concatenate(np.split(grad_attn_output, self.num_heads, axis=2), axis=0)
        # Split the heads
        outputs = np.concatenate(np.split(outputs, self.num_heads, axis=0), axis=-1)
        # Transpose to get original shape
        outputs = np.transpose(outputs, (0, 2, 1, 3))
        # Reshape into (batch_size, seq_len, num_heads, value_size)
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[1], self.num_heads, -1))
        # Sum the gradients across the heads
        grad_attn_output = np.reduce_sum(outputs, axis=2)
        return grad_attn_output
