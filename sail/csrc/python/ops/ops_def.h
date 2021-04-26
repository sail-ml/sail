#pragma once

#include "elementwise.h"
#include "docstring.h"
#include "linalg/linalg.h"
#include "reduction.h"

static PyMethodDef OpsMethods[] = {
    {"add", (PyCFunction)ops_add, METH_VARARGS, NULL},
    {"subtract", (PyCFunction)ops_sub, METH_VARARGS, NULL},
    {"divide", (PyCFunction)ops_div, METH_VARARGS, NULL},
    {"multiply", (PyCFunction)ops_mul, METH_VARARGS, NULL},

    {"reshape", (PyCFunction)ops_reshape, METH_VARARGS, NULL},
    {"expand_dims", (PyCFunction)ops_expand_dims, METH_VARARGS, NULL},

    {"sum", (PyCFunction)ops_sum, METH_VARARGS, NULL},
    {"add_docstring", (PyCFunction)add_docstring, METH_VARARGS, NULL},
    {NULL}};