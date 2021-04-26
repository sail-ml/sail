# from C import *
# from .sail_c import *
from .libsail_c import Tensor
from .libsail_c import add, subtract, divide, multiply
from .libsail_c import reshape, expand_dims
from .libsail_c import sum

__all__ = ["Tensor", "add", "subtract", "divide", "multiply", "reshape", "expand_dims", "sum"]
# import numpy as np

## WRAP TENSORS

# class Tensor():

#     def __new__(cls, arr):
#         arr = np.array(arr)

#         data.__class__ = cls 
#         data.__name__ = cls.__name__