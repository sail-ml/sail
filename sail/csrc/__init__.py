# from C import *
# from .sail_c import *
from .libsail_c import Tensor
from .libsail_c import add, subtract, divide, multiply
from .libsail_c import reshape, expand_dims
from .libsail_c import sum, mean, add_docstring, cast_int32
libsail_c.int32 
print ("Y")
print (libsail_c.int32)
exit()

__all__ = ["Tensor", "add", "subtract", "divide", "multiply", "reshape", "expand_dims", "sum", "mean", "cast_int32"]


