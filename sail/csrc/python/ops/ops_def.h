#pragma once

#include "broadcast.h"
#include "docstring.h"
#include "elementwise.h"
#include "linalg/linalg.h"
#include "math/power.h"
#include "reduction.h"
#include "math/clip.h"
#include "transformations/rollaxis.h"
#include "transformations/transpose.h"

static PyMethodDef OpsMethods[] = {
    {"add", (PyCFunction)ops_add, METH_VARARGS, NULL},
    {"subtract", (PyCFunction)ops_sub, METH_VARARGS, NULL},
    {"divide", (PyCFunction)ops_div, METH_VARARGS, NULL},
    {"multiply", (PyCFunction)ops_mul, METH_VARARGS, NULL},
    {"matmul", (PyCFunction)ops_matmul, METH_VARARGS, NULL},
    {"addmm", (PyCFunction)ops_addmm, METH_VARARGS, NULL},
    {"tensordot", (PyCFunction)ops_tensordot, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"power", (PyCFunction)ops_pow, METH_VARARGS | METH_KEYWORDS, NULL},
    {"exp", (PyCFunction)ops_exp, METH_VARARGS | METH_KEYWORDS, NULL},
    {"log", (PyCFunction)ops_log, METH_VARARGS | METH_KEYWORDS, NULL},

    {"reshape", (PyCFunction)ops_reshape, METH_VARARGS, NULL},
    {"transpose", (PyCFunction)ops_transpose, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"rollaxis", (PyCFunction)ops_rollaxis, METH_VARARGS | METH_KEYWORDS, NULL},
    {"moveaxis", (PyCFunction)ops_moveaxis, METH_VARARGS | METH_KEYWORDS, NULL},
    {"expand_dims", (PyCFunction)ops_expand_dims, METH_VARARGS, NULL},
    {"squeeze", (PyCFunction)ops_squeeze, METH_VARARGS, NULL},

    {"sum", (PyCFunction)ops_sum, METH_VARARGS | METH_KEYWORDS, NULL},
    {"max", (PyCFunction)ops_max, METH_VARARGS | METH_KEYWORDS, NULL},
    {"mean", (PyCFunction)ops_mean, METH_VARARGS | METH_KEYWORDS, NULL},

    {"cast_int32", (PyCFunction)cast_int32, METH_VARARGS, NULL},

    {"add_docstring", (PyCFunction)add_docstring, METH_VARARGS, NULL},

    {"broadcast_to", (PyCFunction)ops_broadcast_to, METH_VARARGS, NULL},
    {"clip", (PyCFunction)ops_clip, METH_VARARGS | METH_KEYWORDS, NULL},

    {NULL}};