from tensor import Tensor
from regularizers import *
import numpy as np


class Layer:
    def __init__(self):
        self.has_weights = False
        self.regularizer = None

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
        return Tensor(self.weights.values[1:, :], requires_grad=True)
    
    def activate(self, X):
        func = getattr(Tensor, self.activation)
        return func(X)


class LayerNorm(Layer):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5
        self.gamma = None
        self.beta = None

    def forward(self, X: Tensor):
        if self.gamma is None and self.beta is None:
            self.gamma = Tensor(np.ones((X.values.shape[1],)), requires_grad=True)
            self.beta = Tensor(np.zeros((X.values.shape[1],)), requires_grad=True)
        return self.gamma * (X.values - X.values.mean(axis=1, keepdims=True)) / (X.values.std(axis=1, keepdims=True) + self.epsilon) + self.beta
    

class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5
        self.gamma = None
        self.beta = None

    def forward(self, X: Tensor):
        if self.gamma is None and self.beta is None:
            self.gamma = Tensor(np.ones((X.values.shape[0],)), requires_grad=True)
            self.beta = Tensor(np.zeros((X.values.shape[0],)), requires_grad=True)
        return self.gamma * (X.values - X.values.mean(axis=0, keepdims=True)) / (X.values.std(axis=0, keepdims=True) + self.epsilon) + self.beta
    
class Dropout(Layer):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, X: Tensor):
        if not (0 <= self.p <= 1):
            raise ValueError("Dropout probability must be in [0, 1)")
        mask = np.random.binomial(1, 1 - self.p, size=X.values.shape)
        return Tensor(X.values * mask, requires_grad=X.requires_grad)