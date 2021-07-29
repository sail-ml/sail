from typing import Any, ClassVar, ModuleType, Type
from typing import overload
from typing import List, Set, Dict, Tuple, Optional

import numpy as np 
import sail
from sail import Tensor

class Loss:
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, *args) -> Tensor: ...

class SoftmaxCrossEntropy(Loss): 
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor: ...

class MeanSquaredError(Loss): 
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor: ...