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
	axis (int or tuple of ints, optional): If provided, then `axis` represents the axis to be summed over. If `axis` is a tuple, then the axes provided will be summed over
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
	axis (int or tuple of ints, optional): If provided, then `axis` represents the axis the mean will be computed over. If `axis` is a tuple, then the axes provided will be used to compute the mean
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
	axis (int or tuple of ints, optional): If provided, then `axis` represents the axis the max will be computed over. If `axis` is a tuple, then the axes provided will be used to compute the max
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
sail.min(tensor, axis=None, keepdims=False) -> Tensor
Returns the minimum of `tensor` over specified axis.

.. note::
	If ``axis < 0``, then the axis that will be computed over is ``tensor.ndim + axis``.

Args:
	tensor (Tensor): Input data
	axis (int or tuple of ints, optional): If provided, then `axis` represents the axis the min will be computed over. If `axis` is a tuple, then the axes provided will be used to compute the min
	keepdims (boolean, optional): If True, then the axes that are reduced will be replaced with 1, otherwise, those axes will be removed

Examples:
	>>> x = sail.random.uniform(0, 1, (12, 32, 4, 5))
	>>> y = sail.min(x, 1, True)
	>>> y.shape
	(12, 1, 4, 5)
	>>> z = sail.min(x, -2, False)
	>>> z.shape
	(12, 32, 5)
	"""
add_docstring(sail.min, descr)

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
	\text{text{out}_{{i}} \sim \mathcal{{N}}(\text{mean}, \text{std})

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
	\text{\text{out}}_{{i}} \sim \mathcal{{N}}(\text{mean}, \text{std})

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
	        [ 1.84346902  0.57650614  0.97455609  1.14918649 -1.60790646]], shape=(5, 5))
	"""
add_docstring(sail.random.normal_like, descr)

descr = r"""
sail.clip(tensor, min, max) -> Tensor
Returns a copy of the tensor with values clipped to between `min` and `max`

Args:
	tensor (Tensor): Input data
	min (float): Min value to compare to
	max (float): Max value to compare to

Examples:
	>>> x = sail.random.normal(5, 3, (10, 10))
	>>> x
	tensor([[ 9.60000324  0.88847959  4.72734308 10.48337078  6.39266825  6.11740732  8.83421326  1.77190900  4.48636198  7.71455336]
	        [ 4.56311178  4.56745386  4.96682215  4.71935797  3.16419554  9.32000637  6.70504379  9.39927197  8.65551376  8.63279819]
	        [ 2.2090044   2.15361404  8.86707401  7.40433693  5.72789907  5.26035881  3.14856267  4.28109598  5.98263168  6.71950102]
	        [ 6.00997019  4.65868711  4.87711191  0.22828352  8.41467190  9.58345127  5.85371304  1.33452487  3.809556    3.11089873]
	        [ 1.22689915  3.28399491  7.02664614  0.05564495  3.01133156  5.09985447  8.34452534  2.40369177  6.7695589   9.7437334 ]
	        [ 2.85278749  7.28451443  3.12173939  4.3741827  -0.09409202  0.15093054  9.33020115  6.44494724  9.04218960  4.82956886]
	        [ 6.7812891   2.04247952  7.07678604  3.05466175  5.28247929  1.32292140  8.3637371   5.92301989  5.43157578  5.7155838 ]
	        [ 3.10557961  0.78700525  4.79287767 -1.29549575 -1.65807295  4.11931086  6.87986612  5.36888313  6.77093649  7.04629421]
	        [ 7.27467346  1.99688959  0.28501093  6.10713482  5.55413008  8.53401089 -2.34596419  8.66580296  7.11808729 14.0457859 ]
	        [10.58966446  2.24703503  6.23982239  1.33525884 10.20398998  6.29514503  8.82054806  4.31791449  4.09643078  3.37176394]], shape=(10, 10))
	>>> sail.clip(x, 4, 6)
	tensor([[6.         4.         4.72734308 6.         6.         6.         6.         4.         4.48636198 6.        ]
	        [4.56311178 4.56745386 4.96682215 4.71935797 4.         6.         6.         6.         6.         6.        ]
	        [4.         4.         6.         6.         5.72789907 5.26035881 4.         4.28109598 5.98263168 6.        ]
	        [6.         4.65868711 4.87711191 4.         6.         6.         5.85371304 4.         4.         4.        ]
	        [4.         4.         6.         4.         4.         5.09985447 6.         4.         6.         6.        ]
	        [4.         6.         4.         4.3741827  4.         4.         6.         6.         6.         4.82956886]
	        [6.         4.         6.         4.         5.28247929 4.         6.         5.92301989 5.43157578 5.7155838 ]
	        [4.         4.         4.79287767 4.         4.         4.11931086 6.         5.36888313 6.         6.        ]
	        [6.         4.         4.         6.         5.55413008 6.         4.         6.         6.         6.        ]
	        [6.         4.         6.         4.         6.         6.         6.         4.31791449 4.09643078 4.        ]], shape=(10, 10))
	"""
add_docstring(sail.clip, descr)

descr = r"""
sail.cat(tensors, axis=0) -> Tensor
Concatenate the `tensors` along axis `axis`

.. note::
	The tensor shapes must match along every axis except `axis`

Args:
	tensors (List of Tensors): Input data
	axis (int): The axis to concatenate on

Examples:
	>>> a = sail.random.normal(2, 3, (2, 3, 4))
	>>> b = sail.random.normal(5, 6, (2, 1, 4))
	>>> sail.cat([a, b], axis=1)
	tensor([[[ 1.55464458  0.8382771   5.41314983  3.65020585]
	        [ 4.19970369 -0.8858959   3.21581888 -2.26835299]
	        [ 7.18340015 -2.02944613  0.24170594  4.67959166]
	        [ 6.80190945 11.18344784  8.07440472  0.44980499]]
	
	        [[ 4.03359985  3.65223217  4.77977037  0.64304084]
	        [ 0.069889   -1.98670769  0.5359674   4.06634569]
	        [ 3.83425260  4.61976576  3.9376111  -1.84104609]
	        [11.87357998  3.23362517  3.74682021  4.41867399]]], shape=(2, 4, 4))
	"""
add_docstring(sail.cat, descr)

descr = r"""
sail.stack(tensors, axis=0) -> Tensor
Stacks the `tensors` by creating a new aixs at `axis`

.. note::
	The tensor shapes must be equal

Args:
	tensors (List of Tensors): Input data
	axis (int): The axis to create

Examples:
	>>> a = sail.random.normal(2, 3, (2, 3, 4))
	>>> b = sail.random.normal(5, 6, (2, 3, 4))
	>>> sail.stack([a,b], axis=1)
	tensor([[[[-1.04555118  5.94666147  3.64661813  4.2072997 ]
	          [-1.6440171  -4.31030703  3.17859864  4.52637434]
	          [ 5.35747242  3.35846686  1.07206750  0.77765495]]
	
	        [[ 5.56071281  6.41274500  7.91134834  2.64334273]
	          [12.57038212  1.11714733 15.22538662 -7.66946459]
	          [ 0.35488090  8.23595142 17.21108818  2.7792344 ]]]
	
	
	        [[[ 3.90003204  3.48069668 -0.62004548 -0.89546055]
	          [ 1.00864995  1.89758968 -0.45129502  6.72634983]
	          [ 2.19042587 -1.64629543  1.35155416  3.7297151 ]]
	
	        [[ 4.43006849  4.48486471  5.26496983 -0.22395246]
	          [15.39132500  6.0622344  -0.05248788 10.19824314]
	          [ 7.6022191   5.0593729  14.05028057  9.58600616]]]], shape=(2, 2, 3, 4))
	"""
add_docstring(sail.stack, descr)
