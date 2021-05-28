import sail 
from sail import add_docstring

# FIND ADD
descr = r"""
sail.add(a, b) -> Tensor
Adds two tensors together using elementwise operations

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
Subtracts two tensors together using elementwise operations

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
Divides two tensors together using elementwise operations

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
Multiplies two tensors together using elementwise operations

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


############## REDUCTIONS ####################

# FIND SUM
descr = r"""
sail.sum(x) -> Tensor
Sums tensor across all axis. DOES NOT SUPPORT AXIS REDUCTION YET

Args:
    x (Tensor): Tensor

Notes:
    DOES NOT SUPPORT AXIS REDUCTION YET

Examples:
    >>> x = sail.random.uniform(0, 1, (10, 50))
    >>> y = sail.sum(x)
    >>> y
    [259.29771532]

"""

add_docstring(sail.sum, descr)


############ LINALG ##################

# FIND MATMUL
descr = r"""
sail.matmul(a, b) -> Tensor
Runs matrix multiplication across two 2D tensors

.. math::
    \text{out} = \text{a} \cdot \text{b}

Args:
    a (Tensor): A 2D tensor
    b (Tensor): A 2D tensor

Notes:
    Inner shapes must match for matrix multiplication 

Examples:
    >>> a = sail.random.uniform(0, 1, (5, 6))
    >>> b = sail.random.uniform(14, 15, (6, 3))
    >>> c = sail.matmul(a, b)
"""

add_docstring(sail.matmul, descr)

# FIND TENSORDOT
descr = r"""
sail.tensordot(a, b, axes=2) -> Tensor
Computes tensor dot product along axes provided

When axes is array_like as ``(a_axes, b_axes)`` then sum the products of 
`a`'s and `b`'s elements over the axis specified in ``(a_axes, b_axes)``.
If axes is an integer `N`, then the last `N` dimensions of `a` and first
`N` dimensions of `b` are used as ``(a_axes, b_axes)``.

Args:
    a (Tensor): A 2D tensor
    b (Tensor): A 2D tensor
    axes (int or (2,) array_like): Axes to be summed over

Notes:
    Common uses:
        * ``axes = 0`` computes tensor product
        * ``axes = 1`` computes tensor dot product
        * ``axes = 2`` computes tensor double contraction

Examples:
    >>> a = sail.random.uniform(0, 1, (3, 4, 5))
    >>> b = sail.random.uniform(14, 15, (4, 3, 2))
    >>> c = sail.tensordot(a, b, axes=([1, 0], [0, 1]))
    >>> c
    tensor([[68.6690377  67.59605111]
            [90.75735783 89.86956470]
            [81.16694957 80.29772401]
            [92.65692908 92.05914529]
            [94.18139766 93.41372834]], shape=(5, 2))

"""

add_docstring(sail.tensordot, descr)

############ MUTATION OPS ##################

# FIND RESHAPE
descr = r"""
sail.reshape(x, new_shape) -> Tensor
Return a reshaped view or copy of x

Args:
    a (Tensor): A tensor
    new_shape (int or tuple of ints): New shape to reshape current tensor to

Notes:
    The product of ``new_shape`` must equal to the product of ``x.shape``.
    Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.

Examples:
    >>> x = sail.random.uniform(10, 20, (5, 10))
    >>> y = sail.reshape(x, (50,))
    >>> y
    tensor([16.18203862 18.50684281 15.42109824 18.73249409 10.97883629 11.32670094 19.33858661 15.04789185 14.09147722 10.3919068 
            11.78628816 11.76781659 18.85068472 15.92606394 13.51610928 16.18784979 10.29533935 15.41951687 13.20227250 16.15049507
            11.9578741  11.72929486 15.90927212 13.98141742 10.31550054 17.21158466 16.4676709  15.0713949  14.1175728  10.878843  
            17.1788167  14.15315004 15.87499473 17.37671597 13.62390671 11.56796764 11.46058571 14.04684157 11.51550815 13.89845898
            15.86720937 12.33080025 19.41888801 17.05646893 18.79913074 18.37086576 13.14104103 18.89630395 13.39037792 11.99629855], shape=(50))
"""

add_docstring(sail.reshape, descr)

# FIND EXPAND_DIMS
descr = r"""
sail.expand_dims(x, axis) -> Tensor
Return a view or copy of x with a dimension inserted at index ``axis``

Args:
    a (Tensor): A tensor
    axis (int): Index to insert a new dimension (can be negative)

Notes:
    Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.

Examples:
    >>> x = sail.random.uniform(10, 20, (5, 10))
    >>> y = sail.expand_dims(x, 0)
    >>> y.shape
    (1, 5, 10)
    >>> y = sail.expand_dims(y, -1)
    >>> y.shape
    (1, 5, 10, 1)
"""

add_docstring(sail.expand_dims, descr)

# FIND SQUEEZE
descr = r"""
sail.squeeze(x, axis) -> Tensor
Return a view or copy of x with a dimension removed at index ``axis``
The dimension size at index ``axis`` must be 1.

Args:
    a (Tensor): A tensor
    axis (int): Index to remove a dimension (can be negative)

Notes:
    Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.

Examples:
    >>> x = sail.random.uniform(10, 20, (5, 1, 10, 1))
    >>> y = sail.squeeze(x, -1)
    >>> y.shape
    (5, 1, 10)
    >>> y = sail.expand_dims(y, 1)
    >>> y.shape
    (5, 10)
"""

add_docstring(sail.squeeze, descr)

# FIND ROLLAXIS
descr = r"""
sail.rollaxis(x, axis, position=0) -> Tensor
Return a view or copy of x with a dimension rolled from index ``axis`` to index ``position``

Args:
    a (Tensor): A tensor
    axis (int): Index to move
    position (int): Index to move ``axis`` to

Notes:
    Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.

Examples:
    >>> x = sail.random.uniform(0, 1, (1, 2, 3, 4, 5))
    >>> y = sail.rollaxis(x, 2, 0)
    >>> y.shape
    (3, 1, 2, 4, 5)
"""

add_docstring(sail.rollaxis, descr)

# FIND TRANSPOSE
descr = r"""
sail.transpose(x, axes=None) -> Tensor
Return a view or copy of x with dimensions reordered to match ``axes``

Args:
    a (Tensor): A tensor
    axis (tuple of ints): New order of axes

Notes:
    Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.

Examples:
    >>> x = sail.random.uniform(2, 4, (2, 3, 4))
    >>> y = sail.transpose(x, (2, 0, 1))
    >>> y.shape
    (4, 2, 3)
"""

add_docstring(sail.transpose, descr)



