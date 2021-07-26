from typing import Any, ClassVar

from typing import overload
import sail
bool_: sail.dtype
float32: sail.dtype
float64: sail.dtype
int16: sail.dtype
int32: sail.dtype
int64: sail.dtype
int8: sail.dtype
uint16: sail.dtype
uint32: sail.dtype
uint64: sail.dtype
uint8: sail.dtype

class DimensionError(sail.SailError): ...

class SailError(Exception): ...

class Tensor:
    grad: ClassVar[getset_descriptor] = ...
    ndim: ClassVar[getset_descriptor] = ...
    requires_grad: ClassVar[getset_descriptor] = ...
    shape: ClassVar[getset_descriptor] = ...
    __hash__: ClassVar[None] = ...
    def __init__(self, *args, **kwargs) -> None: ...
    def astype(self, *args, **kwargs) -> Any: ...
    @overload
    def backward() -> None: ...
    @overload
    def backward() -> Any: ...
    @overload
    def backward() -> Any: ...
    def get_grad(self, *args, **kwargs) -> Any: ...
    def numpy() -> NumpyArray: ...
    def __add__(self, other) -> Any: ...
    def __eq__(self, other) -> Any: ...
    def __ge__(self, other) -> Any: ...
    def __getitem__(self, index) -> Any: ...
    def __gt__(self, other) -> Any: ...
    def __le__(self, other) -> Any: ...
    def __len__(self) -> Any: ...
    def __lt__(self, other) -> Any: ...
    def __mul__(self, other) -> Any: ...
    def __ne__(self, other) -> Any: ...
    def __neg__(self) -> Any: ...
    def __radd__(self, other) -> Any: ...
    def __rmul__(self, other) -> Any: ...
    def __rsub__(self, other) -> Any: ...
    def __rtruediv__(self, other) -> Any: ...
    def __sub__(self, other) -> Any: ...
    def __truediv__(self, other) -> Any: ...

class dtype: ...

@overload
def add(x1, x2) -> Tensor: ...
@overload
def add(a, b) -> Any: ...
def add_docstring(*args, **kwargs) -> Any: ...
def addmm(*args, **kwargs) -> Any: ...
def broadcast_to(tensor, shape) -> Tensor: ...
def cat(tensors, axis = ...) -> Tensor: ...
def clip(tensor, min, max) -> Tensor: ...
@overload
def divide(x1, x2) -> Tensor: ...
@overload
def divide(a, b) -> Any: ...
@overload
def exp(tensor) -> Tensor: ...
@overload
def exp(power) -> Any: ...
def expand_dims(tensor, axis) -> Tensor: ...
@overload
def log(tensor) -> Tensor: ...
@overload
def log(a) -> Any: ...
@overload
def matmul(x1, x2) -> Tensor: ...
@overload
def matmul(a, b) -> Any: ...
def max(tensor, axis = ..., keepdims = ...) -> Tensor: ...
def mean(tensor, axis = ..., keepdims = ...) -> Tensor: ...
def min(tensor, axis = ..., keepdims = ...) -> Tensor: ...
def moveaxis(tensor, axis, position = ...) -> Tensor: ...
@overload
def multiply(x1, x2) -> Tensor: ...
@overload
def multiply(a, b) -> Any: ...
def pad(*args, **kwargs) -> Any: ...
@overload
def power(base, exponent) -> Tensor: ...
@overload
def power(base, exponent) -> Any: ...
def reshape(tensor, shape) -> Tensor: ...
def rollaxis(tensor, axis, position = ...) -> Tensor: ...
def squeeze(tensor, axis) -> Tensor: ...
def stack(tensors, axis = ...) -> Tensor: ...
@overload
def subtract(x1, x2) -> Tensor: ...
@overload
def subtract(a, b) -> Any: ...
def sum(tensor, axis = ..., keepdims = ...) -> Tensor: ...
def tensordot(*args, **kwargs) -> Any: ...
def transpose(tensor, axes) -> Tensor: ...
