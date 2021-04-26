# from .autograd.function import InternalFunction
import numpy as np 
import cupy as cp 
import sail
from typing import Union
from types import ModuleType
from numbers import Number

def get_module(x):
    import numpy, cupy
    # print (type(x))
    # if isinstance(x, numpy.ndarray):
    #     return numpy
    # elif isinstance(x, cupy.ndarray):
    #     return cupy
    # else:
    #     raise TypeError("Type %s not supported" % type(x))
    return cupy.get_array_module(x)


# np_type = ModuleType("numpy")
# cp_type = ModuleType("cupy")

class Tensor():

    # def __init__(self, value, requires_grad=False):
    #     ## TODO: Private
    #     self.value = value 
    #     self.requires_grad = requires_grad

    def __init__(self, data, requires_grad=False, module=None):
        # self = super(Tensor, cls).__new__(cls)
        if isinstance(data, Tensor):
            raise TypeError
        if isinstance(data, tuple):
            data = data[0]


        self.requires_grad = requires_grad
        self.data = data
        if module:
            self.module = module
        else:
            self.module = get_module(data)

        if isinstance(self.data, Number):
            self.data = self.module.asarray(self.data)
        self._shape = self.module.shape(self.data)
        self._size = self.data.size
        # self.dtype = 
        self.grad_fn = None 
        self.grad = None 
        self.has_grad = False

    def __repr__(self):
        start = self.data.__repr__()
        return start

    @property
    def shape(self):
        return self._shape#self.module.shape(self.data)
    @property
    def size(self):
        return self._size#self.module.shape(self.data)

    @property
    def dtype(self):
        return sail.dtype.get_sail_dtype(self.data.dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def T(self):
        ## TODO: IMPLEMENT SWAPAXES FOR BACKPROP
        self.data = self.module.swapaxes(self.data, self.ndim-2, self.ndim-1)
        return self

    def reset_graph(self):
        self.grad_fn = None

    def zero_grad(self):
        self.grad = 0
        self.has_grad = False

    def _cast_data(self, new):
        self.data = self.data.astype(new)
        return self

    def __array_module__(self):
        return get_module(self.data)

    def get_module(self):
        return self.module # NEED TO MAKE SURE THIS IS UPDATEW WHEN DEVICE CHANGES

    def wrap(self, a):
        arr = Tensor(a, requires_grad=self.requires_grad, module=self.module)
        return arr    


    def backward(self, grad=1, depth=1):
        if isinstance(grad, Number):
            grad = Tensor(grad, requires_grad=False, module=self.module)
        if self.has_grad:
            if isinstance(grad.data, tuple):
                grad.data = grad.data[0]
            self.grad += grad.data 
        else:
            if isinstance(grad.data, tuple):
                grad.data = grad.data[0]
            self.grad = grad.data
            self.has_grad = True 

        if self.grad_fn:
            
            incoming = self.grad_fn.ctx.tensors
            grads = self.grad_fn.backward(grad)
            
            if isinstance(grads, tuple):
                for gr, inc in zip(grads, incoming):
                    if inc.requires_grad:
                        inc.backward(gr, depth=depth+1)
            else:
                if incoming[0].requires_grad:
                    incoming[0].backward(grads, depth=depth+1)

            self.reset_graph()


            



Tensor.__add__ = lambda a, b: sail.add(a, b)
Tensor.__radd__ = lambda a, b: sail.add(a, b)

Tensor.__mul__ = lambda a, b: sail.multiply(a, b)
Tensor.__rmul__ = lambda a, b: sail.multiply(a, b)

Tensor.__sub__ = lambda a, b: sail.subtract(a, b)
Tensor.__truediv__ = lambda a, b: sail.divide(a, b)
Tensor.__floordiv__ = lambda a, b: sail.floor_divide(a, b)
Tensor.__pow__ = lambda a, power: sail.power(a, power)
Tensor.__neg__ = lambda a: sail.multiply(a, Tensor(-1))

Tensor.__matmul__ = lambda a, b: sail.matmul(a, b)
