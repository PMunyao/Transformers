#attention mechanism written in jax and stax libraries for 1D data
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax, Softmax

from functools import partial

import numpy as np

import matplotlib.pyplot as plt

import sys
import os

class Attention1D:
    def __init__(self, input_size, hidden_size, output_size, num_heads, batch_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.batch_size = batch_size

        self.build_model()
        self.build_optimizer()

    def build_model(self):
        self.init_random_key = jax.random.PRNGKey(0)
        self.init_random_key, self.init_subkey = jax.random.split(self.init_random_key)
        self.init_random_key, self.init_subkey = jax.random.split(self.init_random_key)
        _, self.init_subkey = jax.random.split(self.init_subkey)

        #create attention weights
        self.A_q = stax.Dense(self.hidden_size, W_init=stax.randn())(self.init_subkey, (self.input_size,))
        self.A_k = stax.Dense(self.hidden_size, W_init=stax.randn())(self.init_subkey, (self.input_size,))
        self.A_v = stax.Dense(self.hidden_size, W_init=stax.randn())(self.init_subkey, (self.input_size,))

        self.A_o = stax.Dense(self.output_size, W_init=stax.randn())(self.init_subkey, (self.hidden_size,))

        #create attention mask
        self.attention_mask = np.zeros((self.batch_size, self.num_heads, self.input_size, self.input_size))
        for i in range(self.batch_size):
            for j in range(self.num_heads):
                self.attention_mask[i,j,:i+1,:i+1] = 1
        self.attention_mask = jnp.array(self.attention_mask)

        #create 6 dense layers
        self.dense_layer_1 = stax.Dense(self.hidden_size, W_init=stax.randn())(self.init_subkey, (self.input_size,))
        self.dense_layer_2 = stax.Dense(self.hidden_size, W_init=stax.randn())(self.init_subkey, (self.hidden_size,))
        self.dense_layer_3 = stax.Dense(self.hidden_size, W_init=stax.randn())(self.init_subkey, (self.hidden_size,))
        self.dense_layer_4 = stax.Dense(self.hidden_size, W_init=stax.randn())(self.init_subkey, (self.hidden_size,))
        self.dense_layer_5 = stax.Dense(self.hidden_size, W_init=stax.randn())(self.init_subkey, (self.hidden_size,))
        self.dense_layer_6 = stax.Dense(self.hidden_size, W_init=stax.randn())(self.init_subkey, (self.hidden_size,))

        #create output layer
        self.output_layer = stax.Dense(self.output_size, W_init=stax.randn())(self.init_subkey, (self.hidden_size,))

        #create function for forward pass
        self.attention_forward = jax.jit(jax.vmap(self.attention_forward_pass, in_axes=(None, 0, 0, 0, 0, 0, 0, 0)))
        self.dense_forward = jax.jit(jax.vmap(self.dense_forward_pass, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        self.output_forward = jax.jit(jax.vmap(self.output_forward_pass, in_axes=(None, 0, 0, 0, 0)))

    def attention_forward_pass(self, x, A_q, A_k, A_v, A_o, attention_mask):
        Q = jnp.dot(A_q, x)
        K = jnp.dot(A_k, x)
        V = jnp.dot(A_v, x)

        Q = jnp.reshape(Q, (self.batch_size, self.num_heads, -1, self.hidden_size))
        K = jnp.reshape(K, (self.batch_size, self.num_heads, -1, self.hidden_size))
        V = jnp.reshape(V, (self.batch_size, self.num_heads, -1, self.hidden_size))

        attention_scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / np.sqrt(self.hidden_size)
        attention_scores = attention_scores + attention_mask
        attention_scores = jnp.reshape(attention_scores, (self.batch_size, -1))
        attention_scores = jax.nn.softmax(attention_scores)
        attention_scores = jnp.reshape(attention_scores, (self.batch_size, self.num_heads, -1, self.input_size))

        attention_output = jnp.matmul(attention_scores, V)
        attention_output = jnp.reshape(attention_output, (self.batch_size, -1, self.hidden_size))

        output = jnp.dot(A_o, attention_output)
        return output

    def dense_forward_pass(self, x, dense_layer_1, dense_layer_2, dense_layer_3, dense_layer_4, dense_layer_5, dense_layer_6, output_layer):
        x = jax.nn.relu(dense_layer_1(x))
        x = jax.nn.relu(dense_layer_2(x))
        x = jax.nn.relu(dense_layer_3(x))
        x = jax.nn.relu(dense_layer_4(x))
        x = jax.nn.relu(dense_layer_5(x))
        x = jax.nn.relu(dense_layer_6(x))
        output = output_layer(x)
        return output

    def output_forward_pass(self, x, output_layer):
        output = output_layer(x)
        return output

    def build_optimizer(self):
        self.learning_rate = 0.001
        self.optimizer = optimizers.adam(step_size=self.learning_rate)

        self.opt_init, self.opt_update, self.get_params = self.optimizer

        self.opt_state = self.opt_init(self.init_params())

        self.update = jax.jit(jax.grad(self.loss))

        @jax.jit
        def update(i, opt_state, batch):
            params = self.get_params(opt_state)
            g = self.update(params, batch)
            return self.opt_update(i, g, opt_state)

        @jax.jit
        def loss(params, batch):
            preds = self.forward_pass(params, batch)
            loss = jnp.mean(jnp.sum(jnp.square(preds - batch[:,1:])))
            return loss

        self.update = update
        self.loss = loss

    def init_params(self):
        _, self.init_subkey = jax.random.split(self.init_subkey)
        _, self.init_subkey = jax.random.split(self.init_subkey)
        _, self.init_subkey = jax.random.split(self.init_subkey)
        _, self.init_subkey = jax.random.split(self.init_subkey)
        _, self.init_subkey = jax.random.split(self.init_subkey)
        _, self.init_subkey = jax.random.split(self.init_subkey)

        A_q = self.A_q.init(self.init_subkey, (self.input_size,))
        A_k = self.A_k.init(self.init_subkey, (self.input_size,))
        A_v = self.A_v.init(self.init_subkey, (self.input_size,))

        A_o = self.A_o.init(self.init_subkey, (self.output_size,))

        dense_layer_1 = self.dense_layer_1.init(self.init_subkey, (self.input_size,))
        dense_layer_2 = self.dense_layer_2.init(self.init_subkey, (self.hidden_size,))
        dense_layer_3 = self.dense_layer_3.init(self.init_subkey, (self.hidden_size,))
        dense_layer_4 = self.dense_layer_4.init(self.init_subkey, (self.hidden_size,))
        dense_layer_5 = self.dense_layer_5.init(self.init_subkey, (self.hidden_size,))
        dense_layer_6 = self.dense_layer_6.init(self.init_subkey, (self.hidden_size,))

        output_layer = self.output_layer.init(self.init_subkey, (self.hidden_size,))

        return (A_q, A_k, A_v, A_o, dense_layer_1, dense_layer_2, dense_layer_3, dense_layer_4, dense_layer_5, dense_layer_6, output_layer)

    def forward_pass(self, params, batch):
        A_q, A_k, A_v, A_o, dense_layer_1, dense_layer_2, dense_layer_3, dense_layer_4, dense_layer_5, dense_layer_6, output_layer = params

        Q = jnp.dot(A_q, batch)
        K = jnp.dot(A_k, batch)
        V = jnp.dot(A_v, batch)

        Q = jnp.reshape(Q, (self.batch_size, self.num_heads, -1, self.hidden_size))
        K = jnp.reshape(K, (self.batch_size, self.num_heads, -1, self.hidden_size))
        V = jnp.reshape(V, (self.batch_size, self.num_heads, -1, self.hidden_size))

        attention_scores = jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / np.sqrt(self.hidden_size)
        attention_scores = attention_scores + self.attention_mask
        attention_scores = jnp.reshape(attention_scores, (self.batch_size, -1))
        attention_scores = jax.nn.softmax(attention_scores)
        attention_scores = jnp.reshape(attention_scores, (self.batch_size, self.num_heads, -1, self.input_size))

        attention_output = jnp.matmul(attention_scores, V)
        attention_output = jnp.reshape(attention_output, (self.batch_size, -1, self.hidden_size))

        x = jnp.dot(A_o, attention_output)
        x = jnp.reshape(x, (self.batch_size, -1, self.output_size))

        x = self.dense_forward(x, dense_layer_1, dense_layer_2, dense_layer_3, dense_layer_4, dense_layer_5, dense_layer_6, output_layer)

        return x

    def train_step(self, batch):
        self.opt_state = self.update(self.opt_state, batch)
        return self.loss(self.get_params(self.opt_state), batch)

    def predict(self, params, x):
        return self.output_forward(params, x)

    def save_weights(self, path):
        path = path + '/weights'
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(path + '/A_q', self.get_params(self.opt_state)[0])
        np.save(path + '/A_k', self.get_params(self.opt_state)[1])
        np.save(path + '/A_v', self.get_params(self.opt_state)[2])
        np.save(path + '/A_o', self.get_params(self.opt_state)[3])
        np.save(path + '/dense_layer_1', self.get_params(self.opt_state)[4])
        np.save(path + '/dense_layer_2', self.get_params(self.opt_state)[5])
        np.save(path + '/dense_layer_3', self.get_params(self.opt_state)[6])
        np.save(path + '/dense_layer_4', self.get_params(self.opt_state)[7])
        np.save(path + '/dense_layer_5', self.get_params(self.opt_state)[8])
        np.save(path + '/dense_layer_6', self.get_params(self.opt_state)[9])
        np.save(path + '/output_layer', self.get_params(self.opt_state)[10])

    def load_weights(self, path):
        path = path + '/weights'
        self.A_q = np.load(path + '/A_q.npy')
        self.A_k = np.load(path + '/A_k.npy')
        self.A_v = np.load(path + '/A_v.npy')
        self.A_o = np.load(path + '/A_o.npy')
        self.dense_layer_1 = np.load(path + '/dense_layer_1.npy')
        self.dense_layer_2 = np.load(path + '/dense_layer_2.npy')
        self.dense_layer_3 = np.load(path + '/dense_layer_3.npy')
        self.dense_layer_4 = np.load(path + '/dense_layer_4.npy')
        self.dense_layer_5 = np.load(path + '/dense_layer_5.npy')
        self.dense_layer_6 = np.load(path + '/dense_layer_6.npy')
        self.output_layer = np.load(path + '/output_layer.npy')

    def load_weights_from_numpy(self, A_q, A_k, A_v, A_o, dense_layer_1, dense_layer_2, dense_layer_3, dense_layer_4, dense_layer_5, dense_layer_6, output_layer):
        self.A_q = A_q
        self.A_k = A_k
        self.A_v = A_v
        self.A_o = A_o
        self.dense_layer_1 = dense_layer_1
        self.dense_layer_2 = dense_layer_2
        self.dense_layer_3 = dense_layer_3
        self.dense_layer_4 = dense_layer_4
        self.dense_layer_5 = dense_layer_5
        self.dense_layer_6 = dense_layer_6
        self.
