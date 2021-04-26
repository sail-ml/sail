#pragma once 

#include "elementwise.h"
#include "reduction.h"
#include "linalg/linalg.h"

static PyMethodDef OpsMethods[] =
{
    { "add", (PyCFunction)ops_add, METH_VARARGS, "add 2 tensors" },
    { "subtract", (PyCFunction)ops_sub, METH_VARARGS, "subtract 2 tensors" },
    { "divide", (PyCFunction)ops_div, METH_VARARGS, "divide 2 tensors" },
    { "multiply", (PyCFunction)ops_mul, METH_VARARGS, "multiply 2 tensors" },

    { "reshape", (PyCFunction)ops_reshape, METH_VARARGS, "reshape" },
    { "expand_dims", (PyCFunction)ops_expand_dims, METH_VARARGS, "expand_dims" },

    { "sum", (PyCFunction)ops_sum, METH_VARARGS, "sum" },
    { NULL }
};