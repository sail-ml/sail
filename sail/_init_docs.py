import sail
from sail import add_docstring


descr = r"""
sail.init.xavier_uniform(x1, gain=1.0) -> Tensor
Fills the tensor `x1` with values generated from an xavier uniform distribution with bounds of ``[-bound, bound)``

.. math::
	\text{bound} = \text{gain} * \sqrt{\frac{6}{fan\_in + fan\_out}}

.. note::
	Xavier uniform method originally described in `Understanding the difficulty of training deep feedforward neural networks <https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_ (Glorot, X. & Bengio, Y., 2010)

.. note::
	Tensor `x1` must have at least 2 dimensions

Args:
	x1 (Tensor): Input tensor to be filled
	gain (float): Scaling parameter


Examples:
	>>> x = sail.random.uniform(0, 1, (10))
	>>> sail.init.xavier_uniform(x, gain=0.5)
	tensor([[-0.06891035 0.24276903 0.33934638 0.15491413]
			[0.16111927 0.36402372 0.36035901 0.39027894]
			[0.23715521 0.31049126 -0.34743783 -0.39470020]
			[0.11064298 -0.00741466 -0.04901789 0.40659216]], shape=(4, 4))
	"""
add_docstring(sail.init.xavier_uniform, descr)

descr = r"""
sail.init.xavier_normal(x1, gain=1.0) -> Tensor
Fills the tensor `x1` with values generated from an xavier normal distribution :math:`\mathcal{{N}}(0, \text{std}^2)`

.. math::
	\text{std} = \text{gain} * \sqrt{\frac{2}{fan\_in + fan\_out}}

.. note::
	Xavier normal method originally described in `Understanding the difficulty of training deep feedforward neural networks <https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_ (Glorot, X. & Bengio, Y., 2010)

.. note::
	Tensor `x1` must have at least 2 dimensions

Args:
	x1 (Tensor): Input tensor to be filled
	gain (float): Scaling parameter


Examples:
	>>> x = sail.random.uniform(0, 1, (10))
	>>> sail.init.xavier_normal(x, gain=0.5)
	tensor([[0.04290007 -0.01282884 0.01678982 0.0718864 ]
			[0.05363203 0.01201495 -0.02504242 -0.04110774]
			[-0.00895256 0.02051808 0.07654259 0.02933300]
			[-0.07215466 0.00211655 0.03904353 -0.0413593 ]], shape=(4, 4))
	"""
add_docstring(sail.init.xavier_normal, descr)

descr = r"""
sail.init.kaiming_uniform(x1, mode="fan_in", nonlin="leaky_relu") -> Tensor
Fills the tensor `x1` with values generated from a kaiming uniform distribution :math:`\mathcal{{U}}(\text{-bound}, \text{bound})`

.. math::
	\text{bound} = gain * \sqrt{\frac{3}{fan\_mode}}

.. note::
	Kaiming uniform method originally described in `Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification <https://arxiv.org/abs/1502.01852>`_ (He, K. et al, 2015)

.. note::
	Tensor `x1` must have at least 2 dimensions

Args:
	x1 (Tensor): Input tensor to be filled
	mode (string): `fan_in` or `fan_out`, where `fan_in` perserves the magnitude of the variance of the weights in the forward pass, where `fan_out` preserves the magnitudes in the backwards pass
	nonlin (string): The non linear activation function used to calculate `gain`, such as `relu` or `leaky_relu`


Examples:
	>>> x = sail.random.uniform(0, 1, (4, 4))
	>>> sail.init.kaiming_uniform(x, "fan_in", "leaky_relu")
	tensor([[-0.50589764 -0.73550594  0.20690525 -0.30652633]
			[-0.24344471 -0.45981696 -0.15996566  1.21492243]
			[-0.97068417  1.18807065  0.69550234  0.19526678]
			[ 0.81102234 -1.03599632  0.71831834 -0.41647190]], shape=(4, 4))
	"""
add_docstring(sail.init.kaiming_uniform, descr)

descr = r"""
sail.init.kaiming_normal(x1, mode="fan_in", nonlin="leaky_relu") -> Tensor
Fills the tensor `x1` with values generated from a kaiming normal distribution :math:`\mathcal{{N}}(0, \text{std}^2)`

.. math::
	\text{std} = \frac{gain}{\sqrt{fan\_mode}}

.. note::
	Kaiming normal method originally described in `Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification <https://arxiv.org/abs/1502.01852>`_ (He, K. et al, 2015)

.. note::
	Tensor `x1` must have at least 2 dimensions

Args:
	x1 (Tensor): Input tensor to be filled
	mode (string): `fan_in` or `fan_out`, where `fan_in` perserves the magnitude of the variance of the weights in the forward pass, where `fan_out` preserves the magnitudes in the backwards pass
	nonlin (string): The non linear activation function used to calculate `gain`, such as `relu` or `leaky_relu`

Examples:
	>>> x = sail.random.uniform(0, 1, (4, 4))
	>>> sail.init.kaiming_normal(x, "fan_in", "leaky_relu")
	tensor([[0.82401055 -0.48167193 -0.48318806 0.64097291]
			[-0.42885232 0.54184997 -0.20086369 -0.87363565]
			[-0.1354739  0.15787299 0.46368173 0.6747992 ]
			[0.1153465  0.9529246  -0.31478310 0.29238516]], shape=(4, 4))
	"""
add_docstring(sail.init.kaiming_normal, descr)