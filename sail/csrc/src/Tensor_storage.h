#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include <iterator>


#include "types.h"
#include "dtypes.h"
#include "error.h"
#include "utils.h"


inline void print(int i) { std::cout << i << " "; }

class TensorStorage {

    public:
        int arr_numel;
        TensorStorage() {};
        TensorStorage(int nd, void* in_data, int64_t in_offset, Dtype dt, std::vector<py::ssize_t> st, std::vector<py::ssize_t> sh) {
            // index = in_index;
            info = getAlignment(dt);

            offset = in_offset;
            dtype = dt;
            ndim = nd;
            strides = st;
            shape = sh;
            arr_numel =  _numel(sh);

            data = _realloc_align(in_data, _numel(sh), info.alignment, info.dtype_size);
           
        }

        void reshape(const TensorSize new_shape) {
            int s = prod_size_vector(new_shape);
            if (s != arr_numel) {
                throw DimensionError{
                    "Cannot reshape tensor of shape ", getVectorString(shape),
                    " to ", getVectorString(new_shape)
                };
            }

            shape = new_shape;
            TensorSize new_strides;
            size_t dt_size = getDtypeSize();
            for (size_t s : shape) {
                new_strides.push_back(dt_size * s);
            }
            new_strides.pop_back();
            new_strides.push_back(dt_size);

            strides = new_strides;
            ndim = shape.size();

        }

        void expand_dims(const int dim) {
            TensorSize s = getShape();
            int pass_dim = 0;
            if (dim == -1) {
                pass_dim = s.size();
            } else {
                pass_dim = dim;
            }
            s.insert(s.begin() + pass_dim, 1);
            reshape(s);
        }

        void free_data() {
            free(data);
        }

        void* operator[](int index) {
            void* x = ((void*)data) + ((index * strides[0]));
            return x;
        }

        TensorSize getShape() {
            return shape;
        }

        static TensorStorage createEmpty(int nd, void* in_data, int64_t in_offset, Dtype dt, std::vector<py::ssize_t> st, std::vector<py::ssize_t> sh) {
            TensorStorage sto;
            sto.info = getAlignment(dt);

            sto.offset = in_offset;
            sto.dtype = dt;
            sto.ndim = nd;
            sto.strides = st;
            sto.shape = sh;
            sto.arr_numel =  _numel(sh);

            sto.data = in_data;
            return sto;
        }

        static TensorStorage createEmpty(int nd, void* in_data, int64_t in_offset, Dtype dt, std::vector<py::ssize_t> st, std::vector<py::ssize_t> sh, alignemnt_information i) {
            TensorStorage sto;
            sto.info = i;

            sto.offset = in_offset;
            sto.dtype = dt;
            sto.ndim = nd;
            sto.strides = st;
            sto.shape = sh;
            sto.arr_numel =  _numel(sh);

            sto.data = in_data;
            return sto;
        }

        // TODO: create a shape class to handle this 
        std::string getShapeString() {
            std::stringstream result;
            std::copy(shape.begin(), shape.end(), std::ostream_iterator<int>(result, ", "));
            std::string x = result.str();
            x.pop_back();
            x.pop_back();
            // std::string  shape_string("(");
            return std::string("(") + x + std::string(")");
        }
        std::string getStridetring() {
            std::string  stride_string("(");
            for (py::ssize_t value : strides) {
                stride_string = stride_string + std::string("aa, ");//("%d, ", value);
            }
            stride_string.pop_back();
            stride_string.pop_back();
            return stride_string + ")";
        }

        std::vector<std::vector<py::ssize_t>> step_back() {
            std::vector<py::ssize_t> new_strides;
            for (int i = 1; i < ndim; i ++) {
                new_strides.push_back(strides[i]);
            }
            std::vector<py::ssize_t> new_shape;
            for (int i = 1; i < ndim; i ++) {
                new_shape.push_back(shape[i]);
            }

            return {new_strides, new_shape};
        }

        std::string getFormat() {
            return GetFormatDescriptor(dtype);
        }
        size_t getDtypeSize() {
            return GetDtypeSize(dtype);
        }
        size_t getTotalSize() {
            auto size = getDtypeSize();
            for (py::ssize_t value : shape) {
                size = size * value;
            }
            return size;
        }

        int numel() const {
            return arr_numel;
        }



        static void check_dimensions_elementwise(TensorStorage a, TensorStorage b) {
            if (a.ndim != b.ndim) {
                throw DimensionError{
                    "Number of dimensions between the two tensors must match. "
                    "Recieved ndims of %d and %d", a.ndim, b.ndim
                };
            } else if (a.shape != b.shape) {
                throw DimensionError{
                    "Shapes between the two tensors must match. "
                    "Recieved shape ", a.getShapeString(), " and ", b.getShapeString()
                };
            }
        }
        
    // private:
        // TensorIndex index;
        int ndim;
        Dtype dtype;

        //Device device;
        void* data;
        int64_t offset;
        std::vector<py::ssize_t> strides;
        TensorSize shape;
        alignemnt_information info;

    private:

        static int _numel(std::vector<py::ssize_t> _shape) {
            auto size = 1;
            for (py::ssize_t value : _shape) {
                size = size * value;
            }
            return size;
        }




};
