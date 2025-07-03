import numpy as np


class Tensor:
    def __init__(self, values, _children=(), _op='', label='', requires_grad=True, retain_values=False):
        self.values = np.array(values, dtype=np.float32)
        self._backward = lambda: None
        self._prev = set(_children)
        self._is_leaf = not _children
        self._op = _op
        self.label = label
        self.requires_grad = requires_grad
        self.retain_values = retain_values
        self.grad = np.zeros_like(self.values) if self.requires_grad else None
        self.epsilon = 1e-10

    def __repr__(self):
        return f"Tensor(values={self.values}, grad={self.grad})"
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.values)

    def detach(self):
        self._prev = set()
        self._op = ''
        self._backward = lambda: None

    def release(self):
        self.values = None if not self.retain_values and not self._is_leaf else self.values


    #Basic methods for forward and backward propagation
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.values + other.values, (self, other), '+')
        def _backward():
            self.grad += out.grad if self.requires_grad else None
            other.grad += out.grad if other.requires_grad else None
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)


    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.values * other.values, (self, other), '*')
            def _backward():
                self.grad += other.values * out.grad if self.requires_grad else None
                other.grad += self.values * out.grad if other.requires_grad else None
            out._backward = _backward
        else:
            out = Tensor(self.values * other, (self,), '*')
            def _backward():
                self.grad += other * out.grad if self.requires_grad else None
            out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.values @ other.values, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.values.T if self.requires_grad else None
            other.grad += self.values.T @ out.grad if other.requires_grad else None
        out._backward = _backward
        return out
    
    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.values @ self.values, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.values.T if self.requires_grad else None
            other.grad += self.values.T @ out.grad if other.requires_grad else None
        out._backward = _backward
        return out
    

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.values - other.values, (self, other), '-')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad 
            if other.requires_grad:
                other.grad -= out.grad
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.values - self.values, (self, other), '-')

        def _backward():
            self.grad -= out.grad if self.requires_grad else None
            other.grad += out.grad if other.requires_grad else None
        out._backward = _backward
        return out
    
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.values / other.values, (self, other), "/")

        def _backward():
            self.grad += out.grad/other.values if self.requires_grad else None
            other.grad -= out.grad*self.values/(other.values*other.values) if other.requires_grad else None
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.values/self.values, (self, other), '/')

        def _backward():
            self.grad -= out.grad*other.values/(self.values*self.values) if self.requires_grad else None
            other.grad += out.grad/self.values if other.requires_grad else None
        out._backward = _backward
        return out
    

    def __pow__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.values ** other.values, (self, other), '**')
            def _backward():
                self.grad += out.grad * other.values * self.values ** (other.values - 1) if self.requires_grad else None
                other.grad += out.grad * out.values * np.log(np.clip(self.values, self.epsilon, None)) if other.requires_grad else None
            out._backward = _backward
            return out
        else:
            out = Tensor(self.values ** other, (self,), f'**{other}')
            def _backward():
                self.grad += out.grad * other * self.values ** (other - 1) if self.requires_grad else None
            out._backward = _backward
            return out

    
    def __rpow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(other.values**self.values, (self, other), '**')

        def _backward():
            self.grad += out.grad * out.values * np.log(np.clip(other.values, self.epsilon, None)) if self.requires_grad else None
            other.grad += out.grad*self.values*other.values**(self.values - 1) if other.requires_grad else None
        out._backward = _backward
        return out
    

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.values > other.values
    
    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.values < other.values
    
    def log(self, base = np.e):
        out = Tensor(np.log(self.values)/np.log(base), (self,), 'log')
        def _backward():
            self.grad += out.grad / np.log(base) / self.values if self.requires_grad else None
        out._backward = _backward
        return out
    
    def log_(self, base=np.e):
        self.values = np.log(self.values) / np.log(base)
        def _backward():
            if self.requires_grad:
                self.grad *= 1 / (np.log(base) * base ** self.values)
        self._backward = _backward
        return self

    
    def exp(self):
        out = Tensor(np.exp(self.values), (self,), 'exp')
        def _backward():
            self.grad += out.grad * out.values if self.requires_grad else None
        out._backward = _backward
        return out
    
    def exp_(self):
        self.values = np.exp(self.values)
        def _backward():
            self.grad *= self.values if self.requires_grad else None
        self._backward = _backward
        return self
    
    def sum(self, axis=None):
        out = Tensor(np.sum(self.values, axis=axis), (self,), 'sum')
        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.values) * out.grad if self.requires_grad else None
            else:
                grad_shape = [1] * self.values.ndim
                grad_shape[axis] = self.values.shape[axis]
                self.grad += np.reshape(out.grad, grad_shape) if self.requires_grad else None
        out._backward = _backward
        return out
    
    def sum_(self, axis=None):
        self.values = np.sum(self.values, axis=axis)
        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.values) if self.requires_grad else None
            else:
                grad_shape = [1] * self.values.ndim
                grad_shape[axis] = self.values.shape[axis]
                self.grad += np.reshape(self.grad, grad_shape) if self.requires_grad else None
        self._backward = _backward
        return self

    def mean(self, axis=None):
        out = Tensor(np.mean(self.values, axis=axis), (self,), 'mean')
        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.values) * out.grad / self.values.size if self.requires_grad else None
            else:
                grad_shape = [1] * self.values.ndim
                grad_shape[axis] = self.values.shape[axis]
                self.grad += np.reshape(out.grad, grad_shape) / self.values.shape[axis] if self.requires_grad else None
        out._backward = _backward
        return out
    
    def mean_(self, axis=None):
        self.values = np.mean(self.values, axis=axis)
        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.values) / self.values.size if self.requires_grad else None
            else:
                grad_shape = [1] * self.values.ndim
                grad_shape[axis] = self.values.shape[axis]
                self.grad += np.reshape(self.grad, grad_shape) / self.values.shape[axis] if self.requires_grad else None
        self._backward = _backward
        return self
    
    def max(self, axis=None):
        out = Tensor(np.max(self.values, axis=axis), (self,), 'max')
        def _backward():
            if axis is None:
                mask = (self.values == out.values)
                self.grad += mask * out.grad if self.requires_grad else None
            else:
                grad_shape = [1] * self.values.ndim
                grad_shape[axis] = self.values.shape[axis]
                mask = (self.values == out.values.reshape(grad_shape))
                self.grad += mask * out.grad.reshape(grad_shape) if self.requires_grad else None
        out._backward = _backward
        return out
    
    def max_(self, axis=None):
        old_values = self.values.copy()
        self.values = np.max(self.values, axis=axis)
        def _backward():
            if axis is None:
                mask = (self.values == old_values)
                self.grad *= mask if self.requires_grad else None
            else:
                grad_shape = [1] * self.values.ndim
                grad_shape[axis] = self.values.shape[axis]
                mask = (self.values == old_values.reshape(grad_shape))
                self.grad *= mask if self.requires_grad else None
        self._backward = _backward
        return self
    
    def min(self, axis=None):
        out = Tensor(np.min(self.values, axis=axis), (self,), 'min')
        def _backward():
            if axis is None:
                mask = (self.values == out.values)
                self.grad += mask * out.grad if self.requires_grad else None
            else:
                grad_shape = [1] * self.values.ndim
                grad_shape[axis] = self.values.shape[axis]
                mask = (self.values == out.values.reshape(grad_shape))
                self.grad += mask * out.grad.reshape(grad_shape) if self.requires_grad else None
        out._backward = _backward
        return out
    
    def min_(self, axis=None):
        old_values = self.values.copy()
        self.values = np.min(self.values, axis=axis)
        def _backward():
            if axis is None:
                mask = (self.values == old_values)
                self.grad *= mask if self.requires_grad else None
            else:
                grad_shape = [1] * self.values.ndim
                grad_shape[axis] = self.values.shape[axis]
                mask = (self.values == old_values.reshape(grad_shape))
                self.grad *= mask if self.requires_grad else None
        self._backward = _backward
        return self

    def abs(self):
        out = Tensor(np.abs(self.values), (self,), 'abs')
        def _backward():
            if self.requires_grad:
                self.grad += np.sign(self.values) * out.grad
        out._backward = _backward
        return out
    
    def abs_(self):
        old_values = self.values.copy()
        self.values = np.abs(self.values)
        def _backward():
            if self.requires_grad:
                self.grad *= np.sign(old_values) 
        self._backward = _backward
        return self
    
    #activation functions
    def softmax(self):
        shifted = self.values - np.max(self.values, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)
        out = Tensor(probs, (self,), 'softmax')
        def _backward():
            s = probs.reshape(-1, 1)
            jacobian = np.diagflat(s) - s @ s.T
            self.grad += (jacobian @ out.grad.reshape(-1,1)).reshape(probs.shape) if self.requires_grad else None
        out._backward = _backward
        return out
    
    def softmax_(self):
        shifted = self.values - np.max(self.values, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        self.values = exps / np.sum(exps, axis=-1, keepdims=True)
        def _backward():
            s = self.values.reshape(-1, 1)
            jacobian = np.diagflat(s) - s @ s.T
            self.grad = (jacobian @ self.grad.reshape(-1,1)).reshape(self.values.shape) if self.requires_grad else None
        self._backward = _backward
        return self


    def sigmoid(self):
        out = Tensor(1/(1 + np.exp(-self.values)), (self,), 'sigmoid')
        def _backward():
            self.grad += out.grad * out.values * (1 - out.values) if self.requires_grad else None
        out._backward = _backward
        return out
    
    def sigmoid_(self):
        self.values = 1/(1 + np.exp(-self.values))
        def _backward():
            self.grad *= self.values * (1 - self.values) if self.requires_grad else None
        self._backward = _backward
        return self

    def tanh(self):
        x = self.values
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t*t) * out.grad if self.requires_grad else None
        out._backward = _backward
        return out
    
    def tanh_(self):
        x = self.values
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        self.values = t
        def _backward():
            self.grad *= (1 - t*t) if self.requires_grad else None
        self._backward = _backward
        return self

    def relu(self):
        out = Tensor(np.maximum(0, self.values), (self,), 'ReLU')
        def _backward():
            self.grad += (out.values > 0) * out.grad if self.requires_grad else None
        out._backward = _backward
        return out
    
    def relu_(self):
        self.values = np.maximum(0, self.values)
        def _backward():
            self.grad *= (self.values > 0) if self.requires_grad else None
        self._backward = _backward
        return self
    
    def linear(self):
        out = Tensor(self.values, (self,), 'linear')
        def _backward():
            self.grad += out.grad if self.requires_grad else None
        out._backward = _backward
        return out
    
    def linear_(self):
        return self

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.values)
        for v in reversed(topo):
            v._backward()
            #v.release()