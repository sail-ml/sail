import sys 

from .libsail import dtype # type: ignore[attr-defined]
from .libsail import Tensor # type: ignore[attr-defined]
from .libsail import add, subtract, divide, multiply, matmul, tensordot, addmm # type: ignore[attr-defined]
from .libsail import reshape, expand_dims, squeeze, clip  # type: ignore[attr-defined]
from .libsail import sum, add_docstring, max, mean, min # type: ignore[attr-defined]
from .libsail import broadcast_to, transpose, rollaxis, moveaxis # type: ignore[attr-defined]
from .libsail import bool_, int8, uint8, int16, uint16, int32, uint32, int64, uint64 # type: ignore[attr-defined]
from .libsail import float32  # type: ignore[attr-defined]
from .libsail import float64  # type: ignore[attr-defined]
from .libsail import cat, stack, pad # type: ignore[attr-defined]
from .libsail import power, exp, log, SailError, DimensionError # type: ignore[attr-defined]

from .libsail import init # type: ignore[attr-defined]
from .libsail import random # type: ignore[attr-defined]
from .libsail import modules # type: ignore[attr-defined]
from .libsail import losses # type: ignore[attr-defined]
from .libsail import optimizers # type: ignore[attr-defined]

sys.modules['sail.init'] = init
sys.modules['sail.random'] = random
sys.modules['sail.modules'] = modules
sys.modules['sail.losses'] = losses
sys.modules['sail.optimizers'] = optimizers

__all__ = ["dtype", "pad", "Tensor", "add", "subtract", "divide", "multiply", "matmul", "reshape", "expand_dims", "squeeze", "sum",
            "broadcast_to", "transpose", "tensordot", "rollaxis", "add_docstring", "bool_", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
           "float32", "float64", "power", "exp", "min", "stack", "cat", "addmm", "max", "mean", "log", "SailError", "DimensionError", "moveaxis", "clip"]

__all__.append("random")
__all__.append("modules")
__all__.append("losses")
__all__.append("optimizers")
__all__.append("init")

