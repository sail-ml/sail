import sail
from sail import add_docstring


descr = r"""
sail.add(x1, x2) -> Tensor
Returns the elementwise addition of Tensors `x1` and `x2`

.. math::
	\text{out} = \text{x1} + \text{x2}

.. note::
	If tensor shapes do not match, they must be broadcastable to a common shape.

Args:
	x1 (Tensor): First Tensor
	x2 (Tensor): Second Tensor

Examples:
	>>> a = sail.random.uniform(0, 1, (20, 3))
	>>> b = sail.random.uniform(14, 15, (20, 3))
	>>> c = sail.add(a, b)
	"""
add_docstring(sail.add, descr)

descr = r"""
sail.subtract(x1, x2) -> Tensor
Returns the elementwise subtraction of Tensor `x2` from `x1`

.. math::
	\text{out} = \text{x1} - \text{x2}

.. note::
	If tensor shapes do not match, they must be broadcastable to a common shape.

Args:
	x1 (Tensor): First Tensor
	x2 (Tensor): Second Tensor

Examples:
	>>> a = sail.random.uniform(0, 1, (20, 3))
	>>> b = sail.random.uniform(14, 15, (20, 3))
	>>> c = sail.subtract(a, b)
	"""
add_docstring(sail.subtract, descr)

descr = r"""
sail.multiply(x1, x2) -> Tensor
Returns the elementwise multiplication of Tensors `x1` and `x2`

.. math::
	\text{out} = \text{x1} * \text{x2}

.. note::
	If tensor shapes do not match, they must be broadcastable to a common shape.

Args:
	x1 (Tensor): First Tensor
	x2 (Tensor): Second Tensor

Examples:
	>>> a = sail.random.uniform(0, 1, (20, 3))
	>>> b = sail.random.uniform(14, 15, (20, 3))
	>>> c = sail.multiply(a, b)
	"""
add_docstring(sail.multiply, descr)

descr = r"""
sail.divide(x1, x2) -> Tensor
Returns the elementwise division of Tensor `x1` by `x2`

.. math::
	\text{out} = \text{x1} / \text{x2}

.. note::
	If tensor shapes do not match, they must be broadcastable to a common shape.

Args:
	x1 (Tensor): First Tensor
	x2 (Tensor): Second Tensor

Examples:
	>>> a = sail.random.uniform(0, 1, (20, 3))
	>>> b = sail.random.uniform(14, 15, (20, 3))
	>>> c = sail.divide(a, b)
	"""
add_docstring(sail.divide, descr)

descr = r"""
sail.power(base, exponent) -> Tensor
Returns Tensor `base` raised to power of Tensor `exponent`.

.. math::
	\text{out} = \text{base}^{\text{power}}

.. note::
	If tensor shapes do not match, they must be broadcastable to a common shape.

Args:
	base (Tensor): Base
	exppnent (Tensor): Exponent

Examples:
	>>> base = sail.random.uniform(0, 1, (20, 3))
	>>> exponent = sail.random.uniform(1, 2, (20, 3))
	>>> c = sail.power(base, exponent)
	"""
add_docstring(sail.power, descr)

descr = r"""
sail.exp(tensor) -> Tensor
Returns e the power `tensor`.

.. math::
	\text{out} = \text{e}^{\text{tensor}}

Args:
	tensor (Tensor): The exponent

Examples:
	>>> power = sail.random.uniform(1, 2, (20, 3))
	>>> b = sail.exp(power)
	"""
add_docstring(sail.exp, descr)

descr = r"""
sail.log(tensor) -> Tensor
Returns natural log of `tensor`

.. math::
	\text{out} = ln(\text{tensor})

Args:
	tensor (Tensor): Input data

Examples:
	>>> a = sail.random.uniform(1, 2, (20, 3))
	>>> b = sail.log(a)
	"""
add_docstring(sail.log, descr)

descr = r"""
sail.sum(tensor, axis=None, keepdims=False) -> Tensor
Returns the sum of `tensor` over specified axis.

.. note::
	If ``axis < 0``, then the axis that will be computed over is ``tensor.ndim + axis``.

Args:
	tensor (Tensor): Input data
	axis (int, optional): If provided, then `axis` represents the axis to be summed over
	keepdims (boolean, optional): If True, then the axes that are reduced will be replaced with 1, otherwise, those axes will be removed

Examples:
	>>> x = sail.random.uniform(0, 1, (12, 32, 4, 5))
	>>> y = sail.sum(x, 1, True)
	>>> y.shape
	(12, 1, 4, 5)
	>>> z = sail.sum(x, -2, False)
	>>> z.shape
	(12, 32, 5)
	"""
add_docstring(sail.sum, descr)

descr = r"""
sail.mean(tensor, axis=None, keepdims=False) -> Tensor
Returns the mean of `tensor` over specified axis.

.. note::
	If ``axis < 0``, then the axis that will be computed over is ``tensor.ndim + axis``.

Args:
	tensor (Tensor): Input data
	axis (int, optional): If provided, then `axis` represents the axis to be summed over
	keepdims (boolean, optional): If True, then the axes that are reduced will be replaced with 1, otherwise, those axes will be removed

Examples:
	>>> x = sail.random.uniform(0, 1, (12, 32, 4, 5))
	>>> y = sail.mean(x, 1, True)
	>>> y.shape
	(12, 1, 4, 5)
	>>> z = sail.mean(x, -2, False)
	>>> z.shape
	(12, 32, 5)
	"""
add_docstring(sail.mean, descr)

descr = r"""
sail.max(tensor, axis=None, keepdims=False) -> Tensor
Returns the maximum of `tensor` over specified axis.

.. note::
	If ``axis < 0``, then the axis that will be computed over is ``tensor.ndim + axis``.

Args:
	tensor (Tensor): Input data
	axis (int, optional): If provided, then `axis` represents the axis to be summed over
	keepdims (boolean, optional): If True, then the axes that are reduced will be replaced with 1, otherwise, those axes will be removed

Examples:
	>>> x = sail.random.uniform(0, 1, (12, 32, 4, 5))
	>>> y = sail.max(x, 1, True)
	>>> y.shape
	(12, 1, 4, 5)
	>>> z = sail.max(x, -2, False)
	>>> z.shape
	(12, 32, 5)
	"""
add_docstring(sail.max, descr)

descr = r"""
sail.matmul(x1, x2) -> Tensor
Returns the matrix multiplication of `x1` and `x2`

.. math::
	\text{out} = \text{x1} \cdot \text{x1}

.. note::
	Both tensor `x1` and `x2` must be 2D, and their inner shapes must match.

Args:
	x1 (Tensor): A 2D Tensor
	x2 (Tensor): A 2D Tensor

Examples:
	>>> a = sail.random.uniform(0, 1, (5, 6))
	>>> b = sail.random.uniform(14, 15, (6, 3))
	>>> c = sail.matmul(a, b)
	"""
add_docstring(sail.matmul, descr)

descr = r"""
sail.tensordot(x1, x2, axes=2) -> Tensor
Returns the tensor dot product along axes provided

When axes is array_like as ``(a_axes, b_axes)`` then sum the products of 
`x1`'s and `x2`'s elements over the axis specified in ``(a_axes, b_axes)``.
If axes is an integer `N`, then the last `N` dimensions of `x1` and first
`N` dimensions of `x2` are used as ``(a_axes, b_axes)``.


.. note::
	Common uses:
    * ``axes = 0`` computes tensor product
    * ``axes = 1`` computes tensor dot product
    * ``axes = 2`` computes tensor double contraction


Args:
	x1 (Tensor): A 2D Tensor
	x2 (Tensor): A 2D Tensor
	axes (int or (2,) array_like): Axes to be summed over

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

descr = r"""
sail.reshape(tensor, shape) -> Tensor
Returns a reshaped view or copy of `tensor`

.. note::
	The product of `shape` must equal to the product of ``tensor.shape``.
Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.


Args:
	tensor (Tensor): Input tensor
	shape (int or tuple of ints): New shape to reshape `tensor` to

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

descr = r"""
sail.expand_dims(tensor, axis) -> Tensor
Returns a view or copy of `tensor` with a dimension inserted at index `axis`

.. note::
	Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.


Args:
	tensor (Tensor): Input tensor
	axis (int): Index to insert new dimension

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

descr = r"""
sail.squeeze(tensor, axis) -> Tensor
Returns a view or copy of `tensor` with a dimension removed at index `axis`

.. note::
	Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.


Args:
	tensor (Tensor): Input tensor
	axis (int): Index to remove dimension

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

descr = r"""
sail.rollaxis(tensor, axis, position=0) -> Tensor
Returns a view or copy of `tensor` with a dimension rolled from index `axis` to index `position`

This differs from ``sail.moveaxis`` because if ``axis > position``, the `axis` is rolled to BEFORE `position`

.. note::
	Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.


Args:
	tensor (Tensor): Input tensor
	axis (int): Index to move
	position (int): Index to `axis` to

Examples:
	>>> x = sail.random.uniform(0, 1, (1, 2, 3, 4, 5))
	>>> y = sail.rollaxis(x, 2, 0)
	>>> y.shape
	(3, 1, 2, 4, 5)
	"""
add_docstring(sail.rollaxis, descr)

descr = r"""
sail.moveaxis(tensor, axis, position=0) -> Tensor
Returns a view or copy of `tensor` with a dimension moved from index `axis` to index `position`

This differs from ``sail.rollaxis`` there is no check for ``axis > position``, which there is in ``sail.rollaxis``

.. note::
	Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.


Args:
	tensor (Tensor): Input tensor
	axis (int): Index to move
	position (int): Index to `axis` to

Examples:
	>>> x = sail.random.uniform(0, 1, (1, 2, 3, 4, 5))
	>>> y = sail.moveaxis(x, 2, 0)
	>>> y.shape
	(3, 1, 2, 4, 5)
	"""
add_docstring(sail.moveaxis, descr)

descr = r"""
sail.transpose(tensor, axes) -> Tensor
Returns a view or copy of `tensor` with dimensions reordered to match `axes`

.. note::
	Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.


Args:
	tensor (Tensor): Input tensor
	axis ((tuple of ints)): New order of axes

Examples:
	>>> x = sail.random.uniform(2, 4, (2, 3, 4))
	>>> y = sail.transpose(x, (2, 0, 1))
	>>> y.shape
	(4, 2, 3)
	"""
add_docstring(sail.transpose, descr)

descr = r"""
sail.broadcast_to(tensor, shape) -> Tensor
Returns a broadcasted view of `tensor`

.. note::
	In order for broadcasting to work, `tensor`'s shape must be compatible with `shape`. Compatability means that, starting from the last shape to the first, the size of the shapes between ``tensor.shape`` and `shape` are either equal, or one of the shapes is 1 or doesn't exist.
Please reference :ref:`view-or-copy` to understand the view and copy semantics of mutation ops.


Args:
	tensor (Tensor): Input tensor
	shape (int or tuple of ints): Shape to broadcast `tensor` to

Examples:
	>>> x = sail.random.uniform(0, 1, (1, 5))
	>>> y = sail.broadcast_to(x, (5, 5))
	>>> y
	tensor([[0.49103028 0.99704099 0.25268090 0.82328194 0.81215674]
	        [0.49103028 0.99704099 0.25268090 0.82328194 0.81215674]
	        [0.49103028 0.99704099 0.25268090 0.82328194 0.81215674]
	        [0.49103028 0.99704099 0.25268090 0.82328194 0.81215674]
	        [0.49103028 0.99704099 0.25268090 0.82328194 0.81215674]], shape=(5, 5))
	"""
add_docstring(sail.broadcast_to, descr)

descr = r"""
sail.random.uniform(min=0, max=1, shape=None) -> Tensor
Returns tensor with the data randomly sampled from a uniform distribution ``[min, max)``

Args:
	min (float, optional): Minimum value of the distribution to sample from
	max (float, optional): Maximum value of the distribution to sample from
	shape (int or tuple of ints, optional): Shape of tensor to generate

Examples:
	>>> x = sail.random.uniform(-1.2, 1.5, (5, 5))
	>>> x
	tensor([[-1.09468198 -0.45043695 -0.24446410 -0.80004257  0.34527856]
	        [-0.11556864  0.2868877   0.72795057  0.10089595 -0.46267119]
	        [ 0.13150010  0.68349499  0.70531654  1.31088471 -0.81466377]
	        [-0.48935455 -0.431037    1.25867462  1.31644297 -0.75355840]
	        [-1.18125939  1.16481411  0.23229218  0.03774609 -0.41145924]], shape=(5, 5))
	"""
add_docstring(sail.random.uniform, descr)

descr = r"""
sail.random.uniform_like(tensor, min=0, max=1) -> Tensor
Returns tensor with the data randomly sampled from a uniform distribution ``[min, max)``, matching the shape and dtype of `tensor`

Args:
	tensor (Tensor): Tensor to pull shape and dtype from
	min (float, optional): Minimum value of the distribution to sample from
	max (float, optional): Maximum value of the distribution to sample from

Examples:
	>>> x = sail.random.uniform(-1.2, 1.5, (5, 5))
	>>> x
	tensor([[-1.09468198 -0.45043695 -0.24446410 -0.80004257  0.34527856]
	        [-0.11556864  0.2868877   0.72795057  0.10089595 -0.46267119]
	        [ 0.13150010  0.68349499  0.70531654  1.31088471 -0.81466377]
	        [-0.48935455 -0.431037    1.25867462  1.31644297 -0.75355840]
	        [-1.18125939  1.16481411  0.23229218  0.03774609 -0.41145924]], shape=(5, 5))
	>>> y = sail.random.uniform_like(x, 0, 1)
	>>> y
	tensor([[0.55789685 0.99225152 0.90981728 0.85239333 0.82039648]
	        [0.01004955 0.49320447 0.71664029 0.22954908 0.06711420]
	        [0.48683277 0.47218278 0.82349455 0.52004343 0.8389914 ]
	        [0.66190428 0.13273992 0.38301027 0.9323185  0.8466584 ]
	        [0.38657394 0.07601264 0.97111505 0.67133880 0.07671827]], shape=(5, 5))
	"""
add_docstring(sail.random.uniform_like, descr)

descr = r"""
sail.random.normal(mean=0, std=1, shape=None) -> Tensor
Returns tensor with the data randomly sampled from a normal distribution

.. math::
	\text{{out}}_{{i}} \sim \mathcal{{N}}(\text{mean}, \text{std})

Args:
	mean (float, optional): Mean of the distribution to sample from
	std (float, optional): Standard deviation of the distribution to sample from
	shape (int or tuple of ints, optional): Shape of tensor to generate

Examples:
	>>> x = sail.random.normal(10, 2, (5, 5))
	>>> x
	tensor([[ 7.74214363  9.92879963 11.89224243 10.95844841 11.03163052]
	        [10.55741215  6.18779612 11.78238773  8.78879356 11.94272995]
	        [11.21028709 14.10308552 11.88035011 13.36896324 10.38013458]
	        [ 6.62501764  4.37194538  8.72444248  7.75856304  9.74470139]
	        [11.36308765  9.17077446 11.09016800  9.5281763  12.16518688]], shape=(5, 5))
	"""
add_docstring(sail.random.normal, descr)

descr = r"""
sail.random.normal_like(tensor, mean=0, std=1) -> Tensor
Returns tensor with the data randomly sampled from a normal distribution, matching the shape and dtype of `tensor`

.. math::
	\text{{out}}_{{i}} \sim \mathcal{{N}}(\text{mean}, \text{std})

Args:
	tensor (Tensor): Tensor to pull shape and dtype from
	mean (float, optional): Mean of the distribution to sample from
	std (float, optional): Standard deviation of the distribution to sample from

Examples:
	>>> x = sail.random.normal(10, 2, (5, 5))
	>>> x
	tensor([[ 7.74214363  9.92879963 11.89224243 10.95844841 11.03163052]
	        [10.55741215  6.18779612 11.78238773  8.78879356 11.94272995]
	        [11.21028709 14.10308552 11.88035011 13.36896324 10.38013458]
	        [ 6.62501764  4.37194538  8.72444248  7.75856304  9.74470139]
	        [11.36308765  9.17077446 11.09016800  9.5281763  12.16518688]], shape=(5, 5))
	>>> y = sail.random.normal_like(x, 0, 1)
	>>> y
	tensor([[ 0.58550960  2.60267472 -2.28640366 -0.33341908 -0.21114956]
	        [-0.67798704 -1.83559990  1.19175482  0.35748199  0.6637082 ]
	        [ 2.05228353  0.42571735 -1.46537054  0.08667278  1.31353283]
	        [ 0.25700086  1.47934365  0.3713915   0.54774326  1.25102699]
	        [ 1.84346902  0.57650614  0.97455609  1.14918649 -1.60790646]], shape=(5, 5))"""
add_docstring(sail.random.normal_like, descr)
