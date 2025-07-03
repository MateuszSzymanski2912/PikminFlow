from tensor import Tensor
import numpy as np


class Dense:
    def __init__(self, dim, input_size = None, activation='linear'):
        self.dim = dim
        self.input_size = input_size
        self.activation = activation
        self.weights = None

    def forward(self, X: Tensor):
        X = Tensor(np.hstack((np.ones((X.values.shape[0], 1)), X.values)))
        return self.activate(X @ self.weights)
    
    def activate(self, X):
        func = getattr(Tensor, self.activation)
        return func(X)