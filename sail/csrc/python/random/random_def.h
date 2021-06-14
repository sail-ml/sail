#pragma once

#include "random_factories.h"

static PyMethodDef RandomFunctions[] = {
    {"uniform", (PyCFunction)ops_random_uniform, METH_VARARGS | METH_KEYWORDS},
    {"uniform_like", (PyCFunction)ops_random_uniform_like,
     METH_VARARGS | METH_KEYWORDS},

    {"normal", (PyCFunction)ops_random_normal, METH_VARARGS | METH_KEYWORDS},
    {"normal_like", (PyCFunction)ops_random_normal_like,
     METH_VARARGS | METH_KEYWORDS},
    {NULL}};