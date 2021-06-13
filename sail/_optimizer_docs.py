import sail 
from sail import add_docstring

def add_docstring_fcn(class_, method, descr):
    cls = getattr(sail.optimizers, class_)
    add_docstring(getattr(cls, method), descr)

def add_docstring_class(class_, descr):
    add_docstring(getattr(sail.optimizers, class_), descr)


add_docstring_class("Optimizer", r"""
Base class for all losses
""")

add_docstring_fcn("Optimizer", "update", r"""
sail.optimizers.Optimizer.update()

Executes the update to parameters defined by the optimizer.
""")
add_docstring_fcn("Optimizer", "track_module", r"""
sail.optimizers.Optimizer.track_module(module)

Gets all parameters from the module and saves them to be tracked, and updated when ``Optimizer.update()`` is called

Args:
    module (Module): Module to extract parameters from

Example:
    >>> linear = sail.modules.Linear(32, 64)
    >>> optimizer = sail.optimizers.SGD(1e-4)
    >>> optimizer.track_module(linear)
""")



add_docstring_class("SGD", r"""
sail.optimizers.SGD(learning_rate)
Executes the following update to parameters tracked by ``Optimizer.track_module(module)``:

.. math::
    \text{updated_param} = \text{param} + -(\text{gradient} * \text(learning\_rate))

Args:
    learning_rate (float): The learning rate that controls the speed of updates
""")
