import sail 
from sail import add_docstring

def add_docstring_fcn(method, descr):
    add_docstring(getattr(sail.Tensor, method), descr)


add_docstring_fcn("numpy", r"""
sail.Tensor.numpy() -> Numpy Array
Returns the numpy representation of the data

This functions returns a copy of the data as a numpy array.

Example:
    >>> x = sail.random.uniform(10, 20, (10, 20))
    >>> x.shape
    (10, 20)
""")

add_docstring_fcn("grad", r"""
Returns the tensor's gradient

If the tensor does not have a gradient or ``requires_grad`` is set to ``False``,
then ``sail.Tensor.grad`` will return ``None``.

Example:
    >>> a = sail.random.uniform(10, 20, (3, 4))
    >>> a.requires_grad = True
    >>> b = sail.random.uniform(10, 20, (3, 4))
    >>> b.requires_grad = True
    >>> c = a + b 
    >>> d = sail.sum(c)
    >>> d.backward()
    >>> a.grad
    tensor([[1. 1. 1. 1.]
            [1. 1. 1. 1.]
            [1. 1. 1. 1.]], shape=(3, 4))
    >>> b.grad
    tensor([[1. 1. 1. 1.]
            [1. 1. 1. 1.]
            [1. 1. 1. 1.]], shape=(3, 4))

""")

add_docstring_fcn("backward", r"""
sail.Tensor.backward() -> None
Run the backpropagation to compute gradients

Whenever you execute an operation on a tensor that requires a gradient, that operation is 
stored on the computation graph. When calling ``backward()`` on a scalar value, the gradients 
are calculated for every tensor and stored in the tensor, which can then be accessed with 
``sail.Tensor.grad``.

.. warning::
    Backwards can only be called on scalar values

Example:
    >>> a = sail.random.uniform(10, 20, (3, 4))
    >>> a.requires_grad = True
    >>> b = sail.random.uniform(10, 20, (3, 4))
    >>> b.requires_grad = True
    >>> c = a + b 
    >>> d = sail.sum(c)
    >>> d.backward()
    >>> a.grad
    tensor([[1. 1. 1. 1.]
            [1. 1. 1. 1.]
            [1. 1. 1. 1.]], shape=(3, 4))
    >>> b.grad
    tensor([[1. 1. 1. 1.]
            [1. 1. 1. 1.]
            [1. 1. 1. 1.]], shape=(3, 4))

""")