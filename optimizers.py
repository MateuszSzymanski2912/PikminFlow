import numpy as np
import inspect


class Optimizer:
    def __init__(self, model):
        self.model = model

    
class SimpleOptimizer(Optimizer):
    name = 'simple'
    def __init__(self, model):
        super().__init__(model)

    def update(self):
        for layer in self.model.layers:
            if layer.has_weights:
                layer.weights.values -= self.model.learning_rate * layer.weights.grad


class MomentumOptimizer(Optimizer):
    name = 'momentum'
    def __init__(self, model, beta = 0.9):
        super().__init__(model)
        self.model = model
        self.beta = beta
        self.m = [np.zeros_like(layer.weights) for layer in self.model.layers]

    def update(self):
        for (l, layer) in enumerate(self.model.layers):
            if layer.has_weights:
                self.m[l] = self.beta * self.m[l] + (1 - self.beta)*layer.weights.grad
                layer.weights.values -= self.model.learning_rate*self.m[l]


class AdaGradOptimizer(Optimizer):
    name = 'adagrad'
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.G = [np.zeros_like(layer.weights) for layer in self.model.layers]

    def update(self):
        for (l, layer) in enumerate(self.model.layers):
            if layer.has_weights:
                self.G[l] += layer.weights.grad**2
                layer.weights.values -= self.model.learning_rate*layer.weights.grad/np.sqrt(self.G[l] + self.epsilon)


class RMSPropOptimizer(Optimizer):
    name = 'rmsprop'
    def __init__(self, model, beta = 0.999):
        super().__init__(model)
        self.model = model
        self.beta = beta
        self.v = [np.zeros_like(layer.weights) for layer in self.model.layers]

    def update(self):
        for (l, layer) in enumerate(self.model.layers):
            if layer.has_weights:
                self.v[l] = self.beta*self.v[l] + (1 - self.beta)*layer.weights.grad**2
                layer.weights.values -= self.model.learning_rate*layer.weights.grad/np.sqrt(self.v[l] + self.epsilon)


class AdamOptimizer(Optimizer):
    name = 'adam'
    def __init__(self, model, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-10):
        super().__init__(model)
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None for layer in self.model.layers]
        self.v = [np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None for layer in self.model.layers]
        self.m_hat = [np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None for layer in self.model.layers]
        self.v_hat = [np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None for layer in self.model.layers]

    def update(self):
        for (l, layer) in enumerate(self.model.layers):
            if layer.has_weights:
                self.m[l] = self.beta1*self.m[l] + (1-self.beta1)*layer.weights.grad
                self.v[l] = self.beta2*self.v[l] + (1-self.beta2)*layer.weights.grad**2
                self.m_hat[l] = self.m[l]/(1-self.beta1**(self.model.epoch + 1))
                self.v_hat[l] = self.v[l]/(1-self.beta2**(self.model.epoch + 1))
                layer.weights.values -= self.model.learning_rate * self.m_hat[l]/(np.sqrt(self.v_hat[l]) + self.epsilon)



def get_optimizer(name):
    for cls in globals().values():
        if inspect.isclass(cls) and issubclass(cls, Optimizer):
            instance_name = getattr(cls, 'name', None)
            if instance_name == name:
                return cls
    raise ValueError(f"No optimizer with name '{name}' found.")
