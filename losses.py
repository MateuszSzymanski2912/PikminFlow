import numpy as np
from tensor import Tensor


def mse(y: Tensor, y_hat: Tensor):
    return ((y-y_hat)*(y-y_hat)).mean()

def mae(y: Tensor, y_hat: Tensor):
      return (y-y_hat).abs().mean()

def binary_crossentropy(y: Tensor, y_hat: Tensor):
     return -(y*y_hat.log() + (1-y)*(1-y_hat).log()).mean()

def categorical_crossentropy(y: Tensor, y_hat: Tensor):
      return -(y*y_hat.log()).mean()

def sparse_categorical_crossentropy(y: Tensor, y_hat: Tensor):
     y_onehot = np.zeros_like(y_hat.values)
     y_onehot[np.arange(len(y.values)), y.values.astype(int)] = 1
     y_onehot = Tensor(y_onehot, requires_grad=False)
     loss = categorical_crossentropy(y_onehot, y_hat)
     return loss


def get_loss(name):
    for func in globals().values():
        if callable(func) and func.__name__ == name:
            return func
    raise ValueError(f"Loss function '{name}' not found.")