
About
================================

SAIL is a python package designed for speed and simplicity when developing and running deep learning models. 
Built on top of a c++ library with python bindings, SAIL is currently in development, changes are being released daily with new features and bug fixes.


What is SAIL
-------------

As long time users of PyTorch and Tensorflow, we decided to venture out and design our own
library that could work just as well as those two, however with some api modifications. The 
result of that is SAIL, which, in early testing, has been shown to be faster than PyTorch in 
certain cases.

What makes it so fast
~~~~~~~~~~~~~~~~~~~~~~~~

The speed of SAIL boils down to three main points

1. C++ Backend

    The C++ backend of SAIL gives the library the ability to do all computations in C++. That gives 
    the library low level instructions that, when compiled, result in massive speed gains over pure 
    python implementations.

2. Python Bindings

    While PyTorch does have Python bindings for its C++ backend, it is very different than how we 
    implement ours. Our bindings are written in such a way that we require very little python for 
    the package to run. In fact, Python only accounts for 18% of the library's codebase, where most 
    of that 18% resides in testing code, as well as docstring code for documentation. These bindings 
    mean that data almost never lives in Python, and all computations are completed in C++, without 
    having to copy data back and forth between the two.

3. Optimizations

    Our C++ backend has numerous optimizations that enable extremely high speeds. For one, we try 
    as hard as possible to never copy data unless it is absolutely necessary. For those who know anything 
    about C++, this is what we mean: Always try to use pass-by-reference, we have implemented our own 
    copy/move semantics, and all tensor data is managed by a pointer, allowing the data to be moved if the 
    container is destroyed. We also make use of SIMD (read: vectorized) operations, allowing for low level 
    cpu instruction vectorization for operations. 