# from C import *
# from .sail_c import *
from .libsail_c import Tensor
from .libsail_c import add, subtract, divide, multiply, matmul, tensordot
from .libsail_c import reshape, expand_dims, squeeze
from .libsail_c import sum, add_docstring, cast_int32
from .libsail_c import broadcast_to, transpose, rollaxis
from .libsail_c import int32 as _int32 
from .libsail_c import float32 as _float32 
from .libsail_c import float64 as _float64 


from .libsail_c import random
from .libsail_c import modules 




# kinda hacky but whatever
int32 = _int32()
float32 = _float32()
float64 = _float64() 

__all__ = ["Tensor", "add", "subtract", "divide", "multiply", "matmul", "reshape", "expand_dims", "squeeze", "sum",
           "int32", "float32", "float64", "broadcast_to", "transpose", "tensordot", "rollaxis", "add_docstring"]

__all__.append("random")
__all__.append("modules")


