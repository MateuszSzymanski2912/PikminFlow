import numpy as np
from tensor import Tensor
from losses import get_loss
from optimizers import get_optimizer
from progress_bar import progress_bar
from layers import *


class Model:
    def __init__(self):
        self.layers = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.model = None
        self.optimizer = None
        self.epoch = 0
        self.batch_size = 0
        self.nodes = None
        self.predictions_history = []
        self.prediction_epochs = []

    def initialize(self, layers, optimizer='adam', batch_size=32, epochs = 100, learning_rate=0.001, loss_function='mse'):
        self.layers = layers
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_indices = np.arange(self.epochs)
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self._init_weights()
        self.optimizer = get_optimizer(self.optimizer)(self)
        self.loss = get_loss(self.loss_function)
        print(f"Model initialized with {len(self.layers)} layers, optimizer: {self.optimizer.name}, batch size: {self.batch_size}, epochs: {self.epochs}, learning rate: {self.learning_rate}, loss function: {self.loss_function}")

    def _init_weights(self):
        if isinstance(self.layers[0], Dense):
            rows = self.layers[0].input_size
            cols = self.layers[0].dim
            self.layers[0].weights = Tensor(np.random.normal(0, np.sqrt(2/rows), (rows+1, cols)))  # Removed +1
            last_layer = 'dense'

        elif isinstance(self.layers[0], Conv2D):
            # Get initial input shape (C, H, W)
            _, C_in, H_in, W_in = self.layers[0].input_size
            K_H, K_W = self.layers[0].kernel_size
            padding = self.layers[0].padding
            stride = self.layers[0].stride
            kernels = self.layers[0].no_of_kernels
            self.layers[0].weights = Tensor(np.random.normal(0, np.sqrt(2/(K_H * K_W * C_in)), (kernels, C_in, K_H, K_W)))
            H_out = (H_in + 2 * padding - K_H) // stride + 1
            W_out = (W_in + 2 * padding - K_W) // stride + 1
            old_kernels = kernels
            last_layer = 'conv2d'

        for layer in self.layers[1:]:
            if isinstance(layer, Dense):
                if last_layer == 'dense':
                    rows = cols
                elif last_layer == 'conv2d':
                    rows = H_out * W_out * old_kernels
                cols = layer.dim
                layer.weights = Tensor(np.random.normal(0, np.sqrt(2/rows), (rows+1, cols)))
                last_layer = 'dense'

            elif isinstance(layer, Conv2D):
                K_H, K_W = layer.kernel_size
                padding = layer.padding
                stride = layer.stride
                kernels = layer.no_of_kernels

                H_out = (H_out + 2 * padding - K_H) // stride + 1
                W_out = (W_out + 2 * padding - K_W) // stride + 1

                layer.weights = Tensor(np.random.normal(0, np.sqrt(2/(K_H * K_W * old_kernels)), (kernels, old_kernels, K_H, K_W)))
                old_kernels = kernels
                last_layer = 'conv2d'


    def forward(self, X: Tensor):
        if not isinstance(X, Tensor):
            X = Tensor(X, requires_grad=False)
        self.nodes = [X]
        for layer in self.layers:
            X = layer.forward(self.nodes[-1])
            self.nodes.append(X)

    def predict(self, X: Tensor):
        if not isinstance(X, Tensor):
            X = Tensor(X, requires_grad=False)
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, loss):
        loss.backward()
        self.optimizer.update()

    def zero_grad(self):
        for layer in self.layers:
            if layer.has_weights:
                layer.weights.zero_grad()
    
    def train_batch(self, X_batch: Tensor, y_batch):
        self.forward(X_batch)
        loss = self.loss(y_batch, self.nodes[-1])
        for layer in self.layers:
            if layer.regularizer and layer.has_weights:
                loss += layer.regularizer(layer.get_weight_no_bias())
        self.backward(loss)
        return loss.values

    def shuffle_data(self, X, y):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        return X_shuffled, y_shuffled
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.epoch = 0
        for epoch in range(self.epochs):
            self.epoch += 1
            epoch_loss = 0
            X_shuffled, y_shuffled = self.shuffle_data(X_train, y_train)
            for i in range(0, len(X_train), self.batch_size):
                self.zero_grad()
                X_batch = Tensor(X_shuffled[i:i+self.batch_size], requires_grad=False)
                y_batch = Tensor(y_shuffled[i:i+self.batch_size], requires_grad=False)
                epoch_loss += self.train_batch(X_batch, y_batch)
            epoch_loss /= len(X_train) / self.batch_size
            if X_val is not None and y_val is not None:
                val_preds = self.predict(Tensor(X_val, requires_grad=False))
                val_loss = self.loss(Tensor(y_val, requires_grad=False), val_preds)
                message = f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {np.round(epoch_loss, 4)}, Validation Loss: {np.round(val_loss.values, 4)}"
                self.history['val_loss'].append(val_loss.values)
            else:
                message = f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {epoch_loss}"
            progress_bar(epoch + 1, self.epochs, message=message, length=30)
            self.history['train_loss'].append(epoch_loss)
        print("\nTraining completed.")
        

    def train_with_animation(self, X, y, x_plot, k = 10):
        self.k = k
        for epoch in range(self.epochs):
            epoch_loss = 0
            X_shuffled, y_shuffled = self.shuffle_data(X, y)
            for i in range(0, len(X), self.batch_size):
                self.zero_grad()
                X_batch = Tensor(X_shuffled[i:i+self.batch_size], requires_grad=False)
                y_batch = Tensor(y_shuffled[i:i+self.batch_size], requires_grad=False)
                epoch_loss += self.train_batch(X_batch, y_batch)
            if (epoch + 1) % k == 0 or epoch == 0 or epoch == self.epochs - 1:
                preds = self.predict(Tensor(x_plot, requires_grad=False))
                self.predictions_history.append(preds.values.flatten())
                self.prediction_epochs.append(epoch + 1)
            progress_bar(epoch + 1, self.epochs, message=f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss}", length=30)
            self.history['train_loss'].append(epoch_loss)
        print("\nTraining completed.")
