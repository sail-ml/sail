from sail import Tensor 
from typing import ClassVar, List, Set, Dict, Tuple, Optional, Type, Union


class Module:
    def __init__(self) -> None: ... 
    def forward(self, x: Tensor) -> Tensor: ... 


class Linear(Module):
    weights: ClassVar[Tensor] = ...
    biases: ClassVar[Tensor] = ...
    def __init__(self, in_features: int, out_features: int, use_bias: bool = True) -> None: ... 
    def forward(self, x: Tensor) -> Tensor: ...
    

class Conv2D(Module):
    weights: ClassVar[Tensor] = ...
    biases: ClassVar[Tensor] = ...
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                     strides: Union[int, Tuple[int]], padding_mode: str = "valid", use_bias: bool = True) -> None: ... 
    def forward(self, x: Tensor) -> Tensor: ...
    

class Sigmoid(Module):
    def __init__(self) -> None: ... 
    def forward(self, x: Tensor) -> Tensor: ...
    

class Softmax(Module):
    def __init__(self) -> None: ... 
    def forward(self, x: Tensor) -> Tensor: ...
    

class ReLU(Module):
    def __init__(self) -> None: ... 
    def forward(self, x: Tensor) -> Tensor: ...
    

class Tanh(Module):
    def __init__(self) -> None: ... 
    def forward(self, x: Tensor) -> Tensor: ...
    