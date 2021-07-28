from .csrc import *
from .csrc import __all__ as __csrc_all__
__all__ = __csrc_all__

import sail.init as init
import sail.random as random
import sail.losses as losses
import sail.optimizers as optimizers
import sail.modules as modules

__all__.append("init")
__all__.append("optimizers")
__all__.append("losses")
__all__.append("random")
__all__.append("modules")

## IMPORT DOCUMENTATION

from ._sail_docs import * # type: ignore[misc]
from ._tensor_docs import * # type: ignore[misc]
from ._module_docs import * # type: ignore[misc]
from ._loss_docs import * # type: ignore[misc]
from ._optimizer_docs import * # type: ignore[misc]
from ._init_docs import * # type: ignore[misc]
