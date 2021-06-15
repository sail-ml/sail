#pragma once

#include "core/dtypes.h"
#include "py_dtype_def.h"
#include "py_dtype_methods.h"



PyDtype* generate_dtype(Dtype dt, int val) {
    PyDtype* new_ = (PyDtype *)PyDtypeBase.tp_alloc(&PyDtypeBase, 0);
    new_->dtype = dt;
    new_->dt_val = val;
    return new_;
}