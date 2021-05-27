import sail 
from sail import add_docstring

# FIND ADD
descr = r"""
sail.add(a, b) -> Tensor
Adds two tensors together using elementwise operations.

.. math::
    \text{out} = \text{a} + \text{b}

Args:
    a (Tensor): First tensor
    b (Tensor): Second tensor

Notes:
    If tensor shapes do not match, they must be broadcastable to a common shape.

Examples:
    >>> a = sail.random.uniform(0, 1, (20, 3))
    >>> b = sail.random.uniform(14, 15, (20, 3))
    >>> c = sail.add(a, b)
"""

add_docstring(sail.add, descr)

# FIND SUBTRACT
descr = r"""
sail.subtract(a, b) -> Tensor
Subtracts two tensors together using elementwise operations.

.. math::
    \text{out} = \text{a} - \text{b}

Args:
    a (Tensor): First tensor
    b (Tensor): Second tensor

Notes:
    If tensor shapes do not match, they must be broadcastable to a common shape.

Examples:
    >>> a = sail.random.uniform(0, 1, (20, 3))
    >>> b = sail.random.uniform(14, 15, (20, 3))
    >>> c = sail.subtract(a, b)
"""

add_docstring(sail.subtract, descr)

# FIND DIVIDE
descr = r"""
sail.divide(a, b) -> Tensor
Divides two tensors together using elementwise operations.

.. math::
    \text{out} = \text{a} / \text{b}

Args:
    a (Tensor): First tensor
    b (Tensor): Second tensor

Notes:
    If tensor shapes do not match, they must be broadcastable to a common shape. 

Examples:
    >>> a = sail.random.uniform(0, 1, (20, 3))
    >>> b = sail.random.uniform(14, 15, (20, 3))
    >>> c = sail.divide(a, b)
"""

add_docstring(sail.divide, descr)

# FIND MULTIPLY
descr = r"""
sail.multiply(a, b) -> Tensor
Multiplies two tensors together using elementwise operations.

.. math::
    \text{out} = \text{a} * \text{b}

Args:
    a (Tensor): First tensor
    b (Tensor): Second tensor

Notes:
    If tensor shapes do not match, they must be broadcastable to a common shape. 

Examples:
    >>> a = sail.random.uniform(0, 1, (20, 3))
    >>> b = sail.random.uniform(14, 15, (20, 3))
    >>> c = sail.multiply(a, b)
"""

add_docstring(sail.multiply, descr)


