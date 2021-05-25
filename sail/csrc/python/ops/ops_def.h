#pragma once

#include "broadcast.h"
#include "docstring.h"
#include "elementwise.h"
#include "linalg/linalg.h"
#include "random/random_factories.h"
#include "reduction.h"
#include "transformations/rollaxis.h"
#include "transformations/transpose.h"

static PyMethodDef OpsMethods[] = {
    {"add", (PyCFunction)ops_add, METH_VARARGS, NULL},
    {"subtract", (PyCFunction)ops_sub, METH_VARARGS, NULL},
    {"divide", (PyCFunction)ops_div, METH_VARARGS, NULL},
    {"multiply", (PyCFunction)ops_mul, METH_VARARGS, NULL},
    {"matmul", (PyCFunction)ops_matmul, METH_VARARGS, NULL},
    {"tensordot", (PyCFunction)ops_tensordot, METH_VARARGS | METH_KEYWORDS,
     NULL},

    {"reshape", (PyCFunction)ops_reshape, METH_VARARGS, NULL},
    {"transpose", (PyCFunction)ops_transpose, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"rollaxis", (PyCFunction)ops_rollaxis, METH_VARARGS | METH_KEYWORDS, NULL},
    {"expand_dims", (PyCFunction)ops_expand_dims, METH_VARARGS, NULL},
    {"squeeze", (PyCFunction)ops_squeeze, METH_VARARGS, NULL},

    {"sum", (PyCFunction)ops_sum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mean", (PyCFunction)ops_mean, METH_VARARGS, NULL},

    {"cast_int32", (PyCFunction)cast_int32, METH_VARARGS, NULL},

    {"add_docstring", (PyCFunction)add_docstring, METH_VARARGS, NULL},

    {"broadcast_to", (PyCFunction)ops_broadcast_to, METH_VARARGS, NULL},

    {NULL}};

static PyMethodDef RandomFactories[] = {
    {"uniform", (PyCFunction)ops_random_uniform, METH_VARARGS | METH_KEYWORDS},
    {"uniform_like", (PyCFunction)ops_random_uniform_like,
     METH_VARARGS | METH_KEYWORDS},
    {NULL}};