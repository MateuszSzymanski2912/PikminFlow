from tensor import Tensor

class Regularizer:
    def __init__(self, strength=0.01):
        self.strength = strength

    def __call__(self, W: Tensor):
        raise NotImplementedError("This method should be overridden by subclasses")
    
class L2(Regularizer):
    def __call__(self, W: Tensor):
        return self.strength * W.square().sum()
    
class L1(Regularizer):
    def __call__(self, W: Tensor):
        return self.strength * W.abs().sum()
    
class ElasticNet(Regularizer):
    def __init__(self, alpha = 0.5):
        self.alpha = alpha
        self.l1_strength = self.strength * alpha
        self.l2_strength = self.strength * (1 - alpha)

    def __call__(self, W: Tensor):
        return (self.l1_strength * W.get_weight_no_bias.abs().sum() +
                self.l2_strength * W.square().sum())