import cupy as cp 
import numpy as np 
import sys
from functools import partial

supported_types = ["float16", "float32", "float64", 
                   "uint8",
                   "int8", "int16", "int32", "int64",
                   "bool"]



class dtype():

    __all__ = supported_types

    @staticmethod
    def get_all():
        return [getattr(sys.modules[__name__], s) for s in supported_types]
    
    @staticmethod
    def create_subclass(name):
        return type(name, (dtype, ), {})

    @staticmethod
    def get_sail_dtype(mod_dtype):
        return getattr(dtype, mod_dtype.name)

    @classmethod
    def get_module_dtype(cls, module):
        return getattr(module, cls.__name__)

# # print (type(__package__))
# print (sys.modules[__name__])
# exit()


# float16 = dtype.create_subclass("float16")
# float32 = dtype.create_subclass("float32")


def generate_all(supported_types):
    out = []
    for s in supported_types:
        sa = type(s, (dtype, ), {
            # constructor
            "__init__": None,
            
            # # data members
            # "string_attribute": "Geeks 4 geeks !",
            # "int_attribute": 1706256,
            
            # # member functions
            # "class_func": classMethod
        })
        out.append([s, sa])
        setattr(dtype, s, sa)

    return out

classes = generate_all(supported_types)
# print (dir(dtype))
# exit()

for c in classes:
    setattr(sys.modules[__name__], c[0], c[1])

__all__ = supported_types

# print (generate_all(supported_types))