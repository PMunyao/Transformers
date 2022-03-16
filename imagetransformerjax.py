#attention mechanism written in jax and stax libraries for 2D image data

import jax.numpy as np
from jax import random
from jax import grad, jit, vmap
from jax import lax
from jax import ops
from jax import jacrev
from jax.experimental import stax
from jax.experimental import optimizers
from jax.experimental import ode

class 2DAttention(object):
    """
    2D attention model using jax and stax libraries
    """
    def __init__(self, input_shape, output_shape, kernel_size,
                 num_filters, batch_size, attention_size,
                 attention_layers, attention_dropout,
                 final_dense_layer,
                 optimizer, learning_rate,
                 use_bias=True,
                 use_edges=True,
                 use_features=True,
                 use_global_features=True,
                 use_attention_dense_layer=True,
                 use_attention_dropout=True):
        """
        Initialize the model
        :param input_shape: Input shape
        :param output_shape: Output shape
        :param kernel_size: Kernel size
        :param num_filters: Number of filters
        :param batch_size: Batch size
        :param attention_size: Attention size
        :param attention_layers: Number of attention layers
        :param attention_dropout: Dropout after attention
        :param final_dense_layer: Use final dense layer
        :param optimizer: Optimizer
        :param learning_rate: Learning rate
        :param use_bias: Use bias
        :param use_edges: Use edges
        :param use_features: Use features
        :param use_global_features: Use global features
        :param use_attention_dense_layer: Use attention dense layer
        :param use_attention_dropout: Use attention dropout
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.attention_size = attention_size
        self.attention_layers = attention_layers
        self.attention_dropout = attention_dropout
        self.final_dense_layer = final_dense_layer
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.use_edges = use_edges
        self.use_features = use_features
        self.use_global_features = use_global_features
        self.use_attention_dense_layer = use_attention_dense_layer
        self.use_attention_dropout = use_attention_dropout
        self.model = self.create_model()

    def create_model(self):
        """
        Create the model
        :return:
        """
        # Input
        inputs = [
            stax.Input(self.input_shape)
        ]

        # Edge convolution
        if self.use_edges:
            inputs.append(
                stax.Input(self.input_shape)
            )

        # Feature maps
        if self.use_features:
            inputs.append(
                stax.Input((self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            )

        # Global features
        if self.use_global_features:
            inputs.append(
                stax.Input((self.input_shape[0], self.input_shape[1], 1))
            )

        # Create the model
        init_random_params, predict = stax.serial(
            # Edge convolution
            stax.Conv(self.num_filters, self.kernel_size,
                      padding='SAME',
                      filter_init=stax.ones,
                      bias_init=stax.normal(0, 1)),
            stax.Relu,
            # Attention
            stax.serial(
                stax.Dense(self.attention_size,
                           filter_init=stax.ones,
                           bias_init=stax.normal(0, 1)),
                stax.Relu,
                stax.Dense(1,
                           filter_init=stax.ones,
                           bias_init=stax.normal(0, 1)),
                stax.Softmax
            ) if self.use_attention_dense_layer else stax.DotProduct(),
            # Final convolution
            stax.Conv(self.num_filters, self.kernel_size,
                      padding='SAME',
                      filter_init=stax.ones,
                      bias_init=stax.normal(0, 1)),
            stax.Relu,
            # Final dense layer
            stax.Dense(self.output_shape[0],
                       filter_init=stax.ones,
                       bias_init=stax.normal(0, 1)) if self.final_dense_layer else stax.serial(
                stax.Flatten,
                stax.Softmax
            ),
        )

        # Create the model
        _, self.params = init_random_params(random.PRNGKey(0), inputs)

        return predict

    def loss(self, X, y):
        """
        Loss function
        :param X: Input
        :param y: Target
        :return: Loss
        """
        # Predict the model
        y_hat = self.model(X, self.params)

        # Compute the loss
        loss = -np.sum(ops.index_update(y, ops.index[:, 1], y_hat) * y) / self.batch_size

        return loss

    def grad(self, X, y):
        """
        Gradient of the loss function
        :param X: Input
        :param y: Target
        :return: Gradient of the loss
        """
        return grad(self.loss)(X, y)

    def accuracy(self, X, y):
        """
        Accuracy of the model
        :param X: Input
        :param y: Target
        :return: Accuracy
        """
        # Predict the model
        y_hat = self.model(X, self.params)

        # Compute the accuracy
        accuracy = np.sum(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1)) / self.batch_size

        return accuracy

    def fit(self, X, y, num_epochs=1000, print_loss=True):
        """
        Fit the model
        :param X: Input
        :param y: Target
        :param num_epochs: Number of epochs
        :param print_loss: Print the loss
        :return:
        """
        # Optimizer
        opt_init, opt_update, get_params = optimizers.sgd(self.learning_rate)

        # Update function
        opt_state = opt_init(self.params)
        update_fn = lambda g: opt_update(g, get_params(opt_state))

        # Train the model
        for epoch in range(num_epochs):
            # Update the parameters
            opt_state = update_fn(self.grad(X, y))

            # Print the loss
            if print_loss and epoch % 100 == 0:
                print('Loss: ' + str(self.loss(X, y)))

        # Save the parameters
        self.params = get_params(opt_state)

    def predict(self, X):
        """
        Predict using the model
        :param X: Input
        :return: Predictions
        """
        return self.model(X, self.params)
        
class Transformers(object):
    """
    Transformers for the 2D attention model
    """
    def __init__(self, input_shape, output_shape, kernel_size, num_filters,
                 use_bias=True,
                 use_edges=True,
                 use_features=True,
                 use_global_features=True,
                 use_attention_dense_layer=True,
                 use_attention_dropout=True):
        """
        Initialize the model
        :param input_shape: Input shape
        :param output_shape: Output shape
        :param kernel_size: Kernel size
        :param num_filters: Number of filters
        :param use_bias: Use bias
        :param use_edges: Use edges
        :param use_features: Use features
        :param use_global_features: Use global features
        :param use_attention_dense_layer: Use attention dense layer
        :param use_attention_dropout: Use attention dropout
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.use_bias = use_bias
        self.use_edges = use_edges
        self.use_features = use_features
        self.use_global_features = use_global_features
        self.use_attention_dense_layer = use_attention_dense_layer
        self.use_attention_dropout = use_attention_dropout
        self.model = self.create_model()

    def create_model(self):
        """
        Create the model
        :return:
        """
        # Input
        inputs = [
            stax.Input(self.input_shape)
        ]

        # Edge convolution
        if self.use_edges:
            inputs.append(
                stax.Input(self.input_shape)
            )

        # Feature maps
        if self.use_features:
            inputs.append(
                stax.Input((self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            )

        # Global features
        if self.use_global_features:
            inputs.append(
                stax.Input((self.input_shape[0], self.input_shape[1], 1))
            )

        # Create the model
        init_random_params, predict = stax.serial(
            # Edge convolution
            stax.Conv(self.num_filters, self.kernel_size,
                      padding='SAME',
                      filter_init=stax.ones,
                      bias_init=stax.normal(0, 1)),
            stax.Relu,
            # Attention
            stax.serial(
                stax.Dense(self.input_shape[0],
                           filter_init=stax.ones,
                           bias_init=stax.normal(0, 1)),
                stax.Relu,
                stax.Dense(1,
                           filter_init=stax.ones,
                           bias_init=stax.normal(0, 1)),
                stax.Softmax
            ) if self.use_attention_dense_layer else stax.DotProduct(),
            # Final convolution
            stax.Conv(self.num_filters, self.kernel_size,
                      padding='SAME',
                      filter_init=stax.ones,
                      bias_init=stax.normal(0, 1)),
            stax.Relu,
            # Final dense layer
            stax.Dense(self.output_shape[0],
                       filter_init=stax.ones,
                       bias_init=stax.normal(0, 1)) if self.final_dense_layer else stax.serial(
                stax.Flatten,
                stax.Softmax
            ),
        )

        # Create the model
        _, self.params = init_random_params(random.PRNGKey(0), inputs)

        return predict

    def loss(self, X, y):
        """
        Loss function
        :param X: Input
        :param y: Target
        :return: Loss
        """
        # Predict the model
        y_hat = self.model(X, self.params)

        # Compute the loss
        loss = -np.sum(ops.index_update(y, ops.index[:, 1], y_hat) * y) / self.batch_size

        return loss

    def grad(self, X, y):
        """
        Gradient of the loss function
        :param X: Input
        :param y: Target
        :return: Gradient of the loss
        """
        return grad(self.loss)(X, y)

    def accuracy(self, X, y):
        """
        Accuracy of the model
        :param X: Input
        :param y: Target
        :return: Accuracy
        """
        # Predict the model
        y_hat = self.model(X, self.params)

        # Compute the accuracy
        accuracy = np.sum(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1)) / self.batch_size

        return accuracy

    def fit(self, X, y, num_epochs=1000, print_loss=True):
        """
        Fit the model
        :param X: Input
        :param y: Target
        :param num_epochs: Number of epochs
        :param print_loss: Print the loss
        :return:
        """
        # Optimizer
        opt_init, opt_update, get_params = optimizers.sgd(self.learning_rate)

        # Update function
        opt_state = opt_init(self.params)
        update_fn = lambda g: opt_update(g, get_params(opt_state))

        # Train the model
        for epoch in range(num_epochs):
            # Update the parameters
            opt_state = update_fn(self.grad(X, y))

            # Print the loss
            if print_loss and epoch % 100 == 0:
                print('Loss: ' + str(self.loss(X, y)))

        # Save the parameters
        self.params = get_params(opt_state)

    def predict(self, X):
        """
        Predict using the model
        :param X: Input
        :return: Predictions
        """
        return self.model(X, self.params)
