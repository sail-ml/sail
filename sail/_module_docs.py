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

add_docstring_class("Conv2D", r"""
sail.modules.Conv2D(in_channels, out_channels, kernel_size, strides, padding_mode="valid", use_bias=True)
Computes Conv2D on data. ONLY DOES SAME PADDING

Args:
    in_channels (int): size of input feature space
    out_channels (int): size of output feature space
    kernel_size (int or tuple): kernel size
    strides (int or tuple): strides
    padding_mode (string): If "valid", no padding is added. "same" will add padding so the output is the same shape as the input
        Default: "valid"
    use_bias (bool): If ``True``, a bias term will be added to the function
        Default: ``True``

Attributes:
    weights: The trainable weights of the operation. The weights are initialized using Kaiming Uniform initialization.
        The shape of the weights is ``(out_channels, in_channels, kernel_height, kernel_width)``.
    bias: The trainable biases of the operation. The biases are initialized to zero, and 
        have a shape of ``(out_channels,)``.
""")
add_docstring_fcn("Conv2D", "forward", "")

add_docstring_class("Sigmoid", r"""
sail.modules.Sigmoid()
Applies the sigmoid function to the input data, which squashes the input to be between [0, 1]

.. math::
    \text{Sigmoid}(x_{i}) = \frac{\mathrm{1} }{\mathrm{1} + e^{-x} }
""")

add_docstring_class("Softmax", r"""
sail.modules.Softmax()
Applies the softmax function to the input data, which squashes the inputs to sum to 1 along dimension 1

.. math::
    \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


""")

add_docstring_class("ReLU", r"""
sail.modules.ReLU()
Applies the relu function to the input data

.. math::
    \text{ReLU}(x) = \max(0, \text{x})

""")

add_docstring_class("Tanh", r"""
sail.modules.Tanh()
Applies the tanh function to the input data

.. math::
    \text{out} = \text{tanh}(x)

""")
