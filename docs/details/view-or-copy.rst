.. _view-or-copy:


View or Copy
==============

When doing operations like ``sail.reshape`` or ``sail.expand_dims``, SAIL will default to returning
a view of tensor being modified. That means when you make a modifications to the actual values of 
the source tensor, the modified tensor will reflect those updates. However, sometimes an operation will 
return a copy. This happens whenever you execute an operation like ``sail.reshape`` or ``sail.expand_dims``
on a view. That will return a copy of the tensor, which has been modified by the function.

So to recap:

* source_tensor -> mutation -> returns view
* source_tensor -> mutation -> mutation -> returns copy 
* source_tensor (view) -> mutation -> returns copy


