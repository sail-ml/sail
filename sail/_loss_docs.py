import sail 
from sail import add_docstring

def add_docstring_fcn(class_, method, descr):
    cls = getattr(sail.losses, class_)
    add_docstring(getattr(cls, method), descr)

def add_docstring_class(class_, descr):
    add_docstring(getattr(sail.losses, class_), descr)


add_docstring_class("Loss", r"""
Base class for all losses
""")

add_docstring_fcn("Loss", "forward", r"""
sail.losses.Loss.forward(*inputs)

Executes the forward propogation defined by the loss.

.. note::
    Call ``Loss(*inputs)`` instead of ``Loss.forward(*inputs)``.
""")



add_docstring_class("SoftmaxCrossEntropy", r"""
sail.losses.SoftmaxCrossEntropy()
Computes the softmax cross entropy loss of input `logits`, compared to `target`

Forward Parameters:
    * **logits** (*Tensor*) – The logits (2D) produced by a model. These are values produced BEFORE a softmax activation
    * **targets** (*Tensor*) – The targets (1D) to compare with. These targets must be integers, where each value indicates the index of the correct class
""")
add_docstring_fcn("SoftmaxCrossEntropy", "forward", "")
