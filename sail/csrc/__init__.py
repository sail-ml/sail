import sys 

from .libsail import Tensor
from .libsail import add, subtract, divide, multiply, matmul, tensordot, addmm
from .libsail import reshape, expand_dims, squeeze, clip
from .libsail import sum, add_docstring, max, mean, min
from .libsail import broadcast_to, transpose, rollaxis, moveaxis
from .libsail import bool_, int8, uint8, int16, uint16, int32, uint32, int64, uint64
from .libsail import float32 
from .libsail import float64 
from .libsail import cat, stack, pad
from .libsail import power, exp, log, SailError, DimensionError

from .libsail import init
from .libsail import random
from .libsail import modules
from .libsail import losses
from .libsail import optimizers

sys.modules['sail.init'] = init
sys.modules['sail.random'] = random
sys.modules['sail.modules'] = modules
sys.modules['sail.losses'] = losses
sys.modules['sail.optimizers'] = optimizers

__all__ = ["pad", "Tensor", "add", "subtract", "divide", "multiply", "matmul", "reshape", "expand_dims", "squeeze", "sum",
            "broadcast_to", "transpose", "tensordot", "rollaxis", "add_docstring", "bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
           "float32", "float64", "power", "exp", "min", "stack", "cat", "addmm", "max", "mean", "log", "SailError", "DimensionError", "moveaxis", "clip"]

__all__.append("random")
__all__.append("modules")
__all__.append("losses")
__all__.append("optimizers")
__all__.append("init")

