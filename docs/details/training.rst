.. _training:


SAIL Model Training
=====================

Building models in SAIL is super simple. There are 3 basic building blocks to creating a model:

* Model Defintion
* Optimizer
* Training Loop

Lets look at these 3 blocks in more depth 

Model Defintion
------------------

SAIL's custom models are built by subclassing the ``sail.modules.Module`` class. This class allows you to worry about only one thing: defining your model.
Things like parameter registration are handled by the parent module, so you don't have to worry about that at all.

Here is a simple example.::

    class MyModel(sail.modules.Module):

        def __init__(self):
            super().__init__()

            self.linear1 = sail.modules.Linear(32, 64)
            self.sigmoid = sail.modules.Sigmoid()
            self.linear2 = sail.modules.Linear(64, 10)
            
        def forward(self, x):
            y = self.linear1(x)
            y = self.sigmoid(y)
            y = self.linear2(y)
            return y 

That's it. This is a simple model that takes an input, passes it through a linear layer, then a sigmoid layer, then another linear layer. But under the hood, 
there is far more happening. When a ``sail.modules.Module`` is subclassed, when instantiated, every attribute is checked to see if it is a module. If it is,
then the parameters are extracted from that module and stored in the parent module's internal parameter tracker. We will see how that is used in the next section,
but essentially, that is how we are allowed to use subclassing as the basis for model building.

Optimizer
------------

So we have defined our module, now we need to attach it to an optimizer. Using simple Stochastic Gradient Descent, lets see what that would look like::

    model = MyModel()
    optimizer = sail.optimizers.SGD(1e-4) # 1e-4 is our learning rate

    optimizer.track_module(model)

Simple right? Before we get an understanding of what is happening, lets see what the full code would look like without subclassing::

    linear1 = sail.modules.Linear(32, 64)
    sigmoid = sail.modules.Sigmoid()
    linear2 = sail.modules.Linear(64, 10)

    def forward(x):
        y = linear1(x)
        y = sigmoid(y)
        y = linear2(y)
        return y 

    optimizer = sail.optimizers.SGD(1e-4) # 1e-4 is our learning rate

    optimizer.track_module(linear1)
    optimizer.track_module(linear2)

See how we would have to track each individual layer (with the exception of Sigmoid because it does not contain any parameters)? 
Imagine how cumbersome that would be with a large model. That is the power of subclassing. Since parameters get rolled into a global
parameter store, we no longer have to worry about attaching each individual layer, we just attach the model itself.

Training Loop 
--------------
    

