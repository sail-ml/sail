import sail 
from sail import add_docstring

def add_docstring_fcn(class_, method, descr):
    cls = getattr(sail.modules, class_)
    add_docstring(getattr(cls, method), descr)

def add_docstring_class(class_, descr):
    add_docstring(getattr(sail.modules, class_), descr)


add_docstring_class("Module", r"""
Base class for all layers
""")

add_docstring_fcn("Module", "forward", r"""
sail.modules.Module.forward(*inputs)

Executes the forward propogation defined by the module.

.. note::
    Call ``Module(*inputs)`` instead of ``Module.forward(*inputs)``.
""")



add_docstring_class("Linear", r"""
sail.modules.Linear(in_features, out_features, use_bias=True)
Computes linear transformation on input data

Args:
    in_features (int): size of input feature space
    out_features (int): size of output feature space
    use_bias (bool): If ``True``, a bias term will be added to the function.
        Default: ``True``

Attributes:
    weights: The trainable weights of the operation. The weights are initialized from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = \frac{1}{\text{in_features}}`.
        The shape of the weights is ``(in_features, out_features)``.
    bias: The trainable biases of the operation. The biases are initialized to zero, and 
        have a shape of ``(out_features,)``.
""")
add_docstring_fcn("Linear", "forward", "")
