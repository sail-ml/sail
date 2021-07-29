from typing import Any, ClassVar, ModuleType, Type
from typing import overload
from typing import List, Set, Dict, Tuple, Optional

import numpy as np 
from sail import Tensor
from sail.modules import Module


class Optimizer:
    def __init__(self) -> None: ...
    def track_module(self, module: Module) -> None: ...
    def update(self) -> None: ...

class SGD(Optimizer): 
    def __init__(self, learning_rate: float) -> Tensor: ...
