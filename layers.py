from tensor import Tensor
from regularizers import *
import numpy as np


class Layer:
    def __init__(self):
        self.has_weights = False
        self.regularizer = None

    def forward(self, X: Tensor):
        raise NotImplementedError("Forward method must be implemented in subclasses")
    
    def __call__(self, X: Tensor):
        return self.forward(X)
    
    def activate(self, X):
        func = getattr(Tensor, self.activation)
        return func(X)

class Dense(Layer):
    def __init__(self, dim, input_size = None, activation='linear', regularizer=None):
        self.dim = dim
        self.input_size = input_size
        self.activation = activation
        self.weights = None
        self.has_weights = True
        self.regularizer = regularizer

    def forward(self, X: Tensor):
        X = Tensor(np.hstack((np.ones((X.values.shape[0], 1)), X.values)))
        return self.activate(X @ self.weights)
    
    def get_weight_no_bias(self):
        return self.weights[1:, :]
    

class Conv2D(Layer):
    def __init__(self, kernel_size, no_of_kernels, input_size = None, stride=1, padding=0, activation='linear', regularizer=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.no_of_kernels = no_of_kernels
        self.input_size = input_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.weights = None
        self.has_weights = True
        self.regularizer = regularizer
        

    def forward(self, X: Tensor):
        if len(X.values.shape) == 4: #Input X should be a 4D tensor (batch_size, channels, height, width)
            (N, C, H, W) = X.values.shape
        elif len(X.values.shape) == 3: #or 3D tensor (batch_size, height, width) -- channels is equal 1 by default then
            (N, H, W) = X.values.shape
            C = 1
        (C_out, _, K_H, K_W) = self.weights.values.shape
        H_out = (H + 2 * self.padding - K_H) // self.stride + 1
        W_out = (W + 2 * self.padding - K_W) // self.stride + 1

        def im2col(X: Tensor):
            cols = []
            X_padded = X.pad(((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
            for i in range(K_H):
                for j in range(K_W):
                    patch = X_padded[:, :, i:i + H_out * self.stride:self.stride, j:j + W_out * self.stride:self.stride]
                    cols.append(patch.reshape((N, C, -1)))
            out = Tensor.stack(cols, axis=2)
            return out.reshape((N, C * K_H * K_W, H_out * W_out))
        
        cols = im2col(X)
        W_col = self.weights.reshape((C_out, -1))
        out = W_col @ cols
        out = out.reshape((N, C_out, H_out, W_out))
        return self.activate(out)
    
    def get_weight_no_bias(self):
        return self.weights


class LayerNorm(Layer):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5
        self.gamma = None
        self.beta = None

    def forward(self, X: Tensor):
        if self.gamma is None and self.beta is None:
            shape = X.values.shape
            param_shape = [1] + list(shape[1:])
            self.gamma = Tensor(np.ones(param_shape), requires_grad=True)
            self.beta = Tensor(np.zeros(param_shape), requires_grad=True)
            
        axis = tuple(i for i in range(X.values.ndim) if i != 0)
        return self.gamma * (X.values - X.values.mean(axis=axis, keepdims=True)) / (X.values.std(axis=axis, keepdims=True) + self.epsilon) + self.beta
    

class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5
        self.gamma = None
        self.beta = None

    def forward(self, X: Tensor):
        if self.gamma is None and self.beta is None:
            shape = X.values.shape
            feature = shape[1]
            param_shape = [1] * len(shape)
            param_shape[1] = feature
            self.gamma = Tensor(np.ones(param_shape), requires_grad=True)
            self.beta = Tensor(np.zeros(param_shape), requires_grad=True)

        axis = tuple(i for i in range(X.values.ndim) if i != 1)
        return self.gamma * (X.values - X.values.mean(axis=axis, keepdims=True)) / (X.values.std(axis=axis, keepdims=True) + self.epsilon) + self.beta
    
class Dropout(Layer):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, X: Tensor):
        if not (0 <= self.p <= 1):
            raise ValueError("Dropout probability must be in [0, 1)")
        mask = np.random.binomial(1, 1 - self.p, size=X.values.shape)
        return Tensor(X.values * mask, requires_grad=X.requires_grad)
    
class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X: Tensor):
        batch_size = X.values.shape[0]
        return X.flatten(batch_size)
