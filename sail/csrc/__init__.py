# from C import *
# from .sail_c import *
from .libsail_c import Tensor
from .libsail_c import add, subtract, divide, multiply, matmul, tensordot, addmm
from .libsail_c import reshape, expand_dims, squeeze, clip
from .libsail_c import sum, add_docstring, max, mean, min
from .libsail_c import broadcast_to, transpose, rollaxis, moveaxis
from .libsail_c import int8, uint8, int16, uint16, int32, uint32, int64, uint64
from .libsail_c import float32 
from .libsail_c import float64 
from .libsail_c import cat, stack
from .libsail_c import power, exp, log, SailError, DimensionError

from .libsail_c import random
from .libsail_c import modules
from .libsail_c import losses
from .libsail_c import optimizers


__all__ = ["Tensor", "add", "subtract", "divide", "multiply", "matmul", "reshape", "expand_dims", "squeeze", "sum",
            "broadcast_to", "transpose", "tensordot", "rollaxis", "add_docstring", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
           "float32", "float64", "power", "exp", "min", "stack", "cat", "addmm", "max", "mean", "log", "SailError", "DimensionError", "moveaxis", "clip"]

__all__.append("random")
__all__.append("modules")
__all__.append("losses")
__all__.append("optimizers")

