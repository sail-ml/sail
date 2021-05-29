import sail 
from sail import add_docstring

def add_docstring_fcn(method, descr):
    add_docstring(getattr(sail.Tensor, method), descr)

print (dir(sail.Tensor))


add_docstring_fcn("numpy", r"""
sail.Tensor.numpy() -> Numpy Array
Returns the numpy representation of the data

This functions returns a copy of the data as a numpy array.

Example:
    >>> x = sail.random.uniform(10, 20, (10, 20))
    >>> x.shape
    (10, 20)
""")