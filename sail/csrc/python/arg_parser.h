#pragma once

#include <Python.h>
// #include <pybind11/pybind11.h>
#include <structmember.h>
#include <chrono>
#include <iostream>
#include "core/Tensor.h"
#include "core/constants.h"
#include "core/dtypes.h"
#include "core/exception.h"
// #include "core/modules/modules.h"
#include "numpy/arrayobject.h"
#include "py_utils.h"

#include <array>
#include <cstddef>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>  // Include for boost::split
#include <boost/algorithm/string/classification.hpp>  // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp>  // Include for boost::split

// namespace py = pybind11;

#define NONE_CHECK_FAIL(v) \
    if (v == Py_None) {    \
        return false;      \
    }
#define NONE_CHECK_PASS(v) \
    if (v == Py_None) {    \
        return true;       \
    }

// namespace sail {

using StringList = typename std::vector<std::string>;
using StringMap = typename std::unordered_map<std::string, std::string>;

enum class ParameterType {
    TENSOR,
    INT64,
    FLOAT,
    TENSOR_LIST,
    INT_LIST,
    BOOL,
    PYOBJECT,
    STRING,
    FLOAT_LIST
};

std::string type_to_name(ParameterType type_) {
    switch (type_) {
        case ParameterType::TENSOR:
            return "Tensor";
        case ParameterType::INT64:
            return "int";
        case ParameterType::FLOAT:
            return "float";
        case ParameterType::TENSOR_LIST:
            return "(tuple of Tensors)";
        case ParameterType::INT_LIST:
            return "(tuple of ints)";
        case ParameterType::FLOAT_LIST:
            return "(tuple of floats)";
        case ParameterType::BOOL:
            return "bool";
        case ParameterType::PYOBJECT:
            return "object";
        case ParameterType::STRING:
            return "str";
        default:
            throw std::runtime_error("unknown parameter type");
            return "object";
    }
    return "object";
}
ParameterType name_to_type(std::string type_) {
    if (type_ == "Tensor") return ParameterType::TENSOR;
    if (type_ == "int") return ParameterType::INT64;
    if (type_ == "float") return ParameterType::FLOAT;
    if (type_ == "TensorList") return ParameterType::TENSOR_LIST;
    if (type_ == "IntList") return ParameterType::INT_LIST;
    if (type_ == "FloatList") return ParameterType::FLOAT_LIST;
    if (type_ == "bool") return ParameterType::BOOL;
    if (type_ == "object") return ParameterType::PYOBJECT;
    if (type_ == "string") return ParameterType::STRING;

    throw std::runtime_error("unknown parameter name " + type_);
    return ParameterType::PYOBJECT;
}

template <int N>
struct FunctionSignature {
    std::string signature;
    std::string function_name;
    StringList arg_order;
    StringList kwarg_order;
    StringList passed_kwarg_order;
    StringMap args;
    StringMap args_types;
    int arg_count;

    int size_args = 0;
    int size_kwargs = 0;

    FunctionSignature(std::string sig) {
        signature = sig;

        StringList split;
        boost::split(split, signature, boost::is_any_of("("),
                     boost::token_compress_on);
        function_name = split[0];
        sig = split[1];

        sig = std::regex_replace(sig, std::regex("\\)"), "");

        StringList args_;
        boost::split(args_, sig, boost::is_any_of(","),
                     boost::token_compress_on);

        for (std::string arg : args_) {
            StringList arg_and_type;
            boost::algorithm::trim(arg);
            boost::split(arg_and_type, arg, boost::is_any_of(" "),
                         boost::token_compress_on);
            std::string arg_no_extra = arg_and_type[1];
            std::string type = arg_and_type[0];
            boost::algorithm::trim(arg_no_extra);
            if (boost::algorithm::contains(arg, "=")) {
                StringList arg_with_default;
                boost::split(arg_with_default, arg, boost::is_any_of("="),
                             boost::token_compress_on);
                boost::algorithm::trim(arg_with_default[1]);
                size_kwargs += 1;
                if (type == "string") {
                    arg_with_default[1] = std::regex_replace(
                        arg_with_default[1], std::regex("\""), "");
                    arg_with_default[1] = std::regex_replace(
                        arg_with_default[1], std::regex("\'"), "");
                }
                args.insert({arg_no_extra, arg_with_default[1]});
                kwarg_order.push_back(arg_no_extra);
            } else {
                args.insert({arg_no_extra, ""});
                size_args += 1;
            }
            arg_order.push_back(arg_no_extra);
            args_types.insert({arg_no_extra, type});
        }
        arg_count = args.size();
    }

    std::vector<PyObject*> merge(PyObject* args, PyObject* kwargs, int max_n) {
        std::vector<PyObject*> py_args(N);

        int actual_args_size = PyTuple_Size(args);
        if (actual_args_size < size_args) {
            THROW_ERROR_DETAILED(TypeError,
                                 "Not enough positional arguments provided");
        }
        int max = size_args + size_kwargs;

        int i = 0;
        for (i = 0; i < actual_args_size; i++) {
            py_args[i] = PyTuple_GET_ITEM(args, i);
        }
        if (size_kwargs && i < max && kwargs != NULL) {
            int st = i;
            int diff = actual_args_size - size_args;
            for (i = diff; i < size_kwargs; i++) {
                auto k = kwarg_order[i];
                PyObject* val =
                    PyDict_GetItem(kwargs, PyUnicode_FromString(k.c_str()));
                if (val != NULL) {
                    py_args[st] = val;
                } else {
                    py_args[st] = Py_None;
                }
                st += 1;
            }
        }
        return py_args;
    }

    std::string generate_error_text() {
        auto out = function_name + "(";
        for (std::string arg : arg_order) {
            auto arg_type = args_types[arg];
            out += type_to_name(name_to_type(arg_type)) + " " + arg + ", ";
        }
        out.pop_back();
        out.pop_back();
        out += ")";
        return out;
    }

    bool parse(std::vector<PyObject*> passed_args, StringList passed_in_names) {
        bool is_good = false;
        int args_actually_found = 0;
        int count_kwarg = 0;

        if (passed_args.size() != arg_count) {
            return false;
        }

        int c_ = 0;
        for (int i = 0; i < passed_in_names.size(); i++) {
            auto in_name = passed_in_names[i];
            auto check_name = arg_order[c_];
            if (in_name != check_name && in_name != "") {
                if (args[check_name] == "") {
                    return false;
                }
                c_ -= 1;
            }
            c_ += 1;
        }

        for (int i = 0; i < N; i++) {
            std::string arg_name = arg_order[i];
            std::string type = args_types[arg_name];
            std::string default_val = args[arg_name];

            PyObject* arg = passed_args[i];
            if (arg == NULL || arg == nullptr) {
                // return is_good;
                continue;
            }
            if (arg == Py_None) {
                is_good = true;
            }

            ParameterType type_ = name_to_type(type);

            if (default_val == "") {
                args_actually_found += 1;
            }
            auto allow_none = false;
            if (default_val != "") {
                allow_none = true;
            }

            try {
                switch (type_) {
                    case ParameterType::INT64:
                        is_good = (verify_int(
                            arg, allow_none));  // || (default_val == "None"));
                        break;
                    case ParameterType::FLOAT:
                        is_good = (verify_float(
                            arg, allow_none));  // || (default_val == "None"));
                        break;
                    case ParameterType::TENSOR:
                        is_good = (verify_Tensor(
                            arg, allow_none));  // || (default_val == "None"));
                        break;
                    case ParameterType::INT_LIST:
                        is_good = (verify_int_list(
                            arg, allow_none));  // || (default_val == "None"));
                        break;
                    case ParameterType::FLOAT_LIST:
                        is_good = (verify_float_list(
                            arg, allow_none));  // || (default_val == "None"));
                        break;
                    case ParameterType::TENSOR_LIST:
                        is_good = (verify_tensor_list(
                            arg, allow_none));  // || (default_val != "None"));
                        break;
                    case ParameterType::BOOL:
                        is_good = (verify_bool(
                            arg, allow_none));  // || (default_val == "None"));
                        break;
                    case ParameterType::STRING:
                        is_good = (verify_string(
                            arg, allow_none));  // || (default_val == "None"));
                        break;
                    case ParameterType::PYOBJECT:
                        is_good = true;
                        break;

                    default:
                        throw std::runtime_error("Type not supported");
                        break;
                }

                if (!is_good) {
                    return false;
                }
            } catch (std::exception& e) {
                return false;
            }
        }
        if (args_actually_found != size_args) {
            return false;
        }

        return true;
    };

    bool verify_int(PyObject* obj, bool allow_none) {
        if (!allow_none) {
            NONE_CHECK_FAIL(obj);
        } else {
            NONE_CHECK_PASS(obj);
        }
        return PyLong_Check(obj);
    }
    bool verify_float(PyObject* obj, bool allow_none) {
        if (!allow_none) {
            NONE_CHECK_FAIL(obj);
        } else {
            NONE_CHECK_PASS(obj);
        }
        return PyFloat_Check(obj) || PyLong_Check(obj);
    }
    bool verify_Tensor(PyObject* obj, bool allow_none) {
        if (!allow_none) {
            NONE_CHECK_FAIL(obj);
        } else {
            NONE_CHECK_PASS(obj);
        }
        return ((obj)->ob_type == (&PyTensorType) ||
                PyType_IsSubtype((obj)->ob_type, (&PyTensorType)));
    }
    bool verify_bool(PyObject* obj, bool allow_none) {
        if (!allow_none) {
            NONE_CHECK_FAIL(obj);
        } else {
            NONE_CHECK_PASS(obj);
        }
        return PyBool_Check(obj);
    }
    bool verify_string(PyObject* obj, bool allow_none) {
        if (!allow_none) {
            NONE_CHECK_FAIL(obj);
        } else {
            NONE_CHECK_PASS(obj);
        }
        return PyUnicode_Check(obj);
    }
    bool verify_int_list(PyObject* obj, bool allow_none) {
        if (!allow_none) {
            NONE_CHECK_FAIL(obj);
        } else {
            NONE_CHECK_PASS(obj);
        }
        auto tuple = PyTuple_Check(obj);
        auto list = PyList_Check(obj);

        if (!tuple && !list) {
            return false;
        }

        auto size = tuple ? PyTuple_Size(obj) : PyList_Size(obj);
        if (size < 0) {
            return false;
        }
        for (int i = 0; i < size; i++) {
            PyObject* check =
                tuple ? PyTuple_GetItem(obj, i) : PyList_GetItem(obj, i);
            if (!verify_int(check, allow_none)) {
                return false;
            }
        }
        return true;
    }
    bool verify_float_list(PyObject* obj, bool allow_none) {
        if (!allow_none) {
            NONE_CHECK_FAIL(obj);
        } else {
            NONE_CHECK_PASS(obj);
        }
        auto tuple = PyTuple_Check(obj);
        auto list = PyList_Check(obj);

        if (!tuple && !list) {
            return false;
        }

        auto size = tuple ? PyTuple_Size(obj) : PyList_Size(obj);
        if (size < 0) {
            return false;
        }
        for (int i = 0; i < size; i++) {
            PyObject* check =
                tuple ? PyTuple_GetItem(obj, i) : PyList_GetItem(obj, i);
            if (!verify_float(check, allow_none)) {
                return false;
            }
        }
        return true;
    }
    bool verify_tensor_list(PyObject* obj, bool allow_none) {
        if (!allow_none) {
            NONE_CHECK_FAIL(obj);
        } else {
            NONE_CHECK_PASS(obj);
        }
        auto tuple = PyTuple_Check(obj);
        auto list = PyList_Check(obj);

        if (!tuple && !list) {
            return false;
        }

        auto size = tuple ? PyTuple_Size(obj) : PyList_Size(obj);
        if (size < 0) {
            return false;
        }
        for (int i = 0; i < size; i++) {
            PyObject* check =
                tuple ? PyTuple_GetItem(obj, i) : PyList_GetItem(obj, i);
            if (!verify_Tensor(check, allow_none)) {
                return false;
            }
        }
        return true;
    }

    bool has_default(int i) {
        std::string arg_name = arg_order[i];
        std::string type = args_types[arg_name];
        std::string default_val = args[arg_name];
        if (default_val != "") {
            return true;
        }
        return false;
    }

    int get_default_int(int i) {
        std::string arg_name = arg_order[i];
        std::string type = args_types[arg_name];
        std::string default_val = args[arg_name];

        if (default_val == "None") {
            return 0;
        }

        return atoi(default_val.c_str());
    }
    std::vector<long> get_default_int_list(int i) {
        std::string arg_name = arg_order[i];
        std::string type = args_types[arg_name];
        std::string default_val = args[arg_name];

        std::vector<long> out = {};

        return out;
    }
    int get_default_int_or_axis(int i) {
        std::string arg_name = arg_order[i];
        std::string type = args_types[arg_name];
        std::string default_val = args[arg_name];
        if (default_val == "None") {
            return NULLDIM;
        }

        return atoi(default_val.c_str());
    }
    double get_default_double(int i) {
        std::string arg_name = arg_order[i];
        std::string type = args_types[arg_name];
        std::string default_val = args[arg_name];

        return atof(default_val.c_str());
    }
    double get_default_bool(int i) {
        std::string arg_name = arg_order[i];
        std::string type = args_types[arg_name];
        std::string default_val = args[arg_name];

        return (default_val == "true" || default_val == "True");
    }
    std::string get_default_string(int i) {
        std::string arg_name = arg_order[i];
        std::string type = args_types[arg_name];
        std::string default_val = args[arg_name];

        return default_val;
    }
};

template <int N>  // max number of params across all signatures
struct PythonArgParser {
    std::vector<std::vector<PyObject*>> args;
    // PyObject* args[N];
    PyObject* in_args;
    PyObject* in_kwargs;
    StringList passed_arg_list_names;

    std::vector<bool> matches;
    int use = -1;

    std::vector<FunctionSignature<N>> signatures;

    PythonArgParser(std::vector<std::string> signatures_, PyObject* _in_args,
                    PyObject* _in_kwargs) {
        for (std::string sig : signatures_) {
            signatures.push_back(FunctionSignature<N>(sig));
        }
        in_args = _in_args;
        in_kwargs = _in_kwargs;
        merge();
    }

    void merge() {
        for (auto& s : signatures) {
            auto res = s.merge(in_args, in_kwargs, N);
            args.push_back(res);
        }
        if (in_args) {
            int si = PyTuple_Size(in_args);
            for (int i = 0; i < si; i++) {
                passed_arg_list_names.push_back("");
            }
        }
        if (in_kwargs) {
            auto list = PyDict_Keys(in_kwargs);
            int si = PyList_Size(list);
            for (int i = 0; i < si; i++) {
                auto name = PyUnicode_AsUTF8(PyList_GetItem(list, i));
                passed_arg_list_names.push_back(name);
            }
        }
    }

    bool parse() {
        std::string sig_strings;
        int i = 0;
        for (auto sig : signatures) {
            sig_strings += sig.generate_error_text() + "\n";
            auto result = sig.parse(args[i], passed_arg_list_names);
            i += 1;
            matches.push_back(result);
        }
        i = 0;
        for (bool m : matches) {
            if (m == true) {
                use = i;
                return true;
            }
            i += 1;
        }
        THROW_ERROR_DETAILED(
            TypeError, "Arguments do not match defined function signatures: \n",
            sig_strings);
        return false;
    }

    bool at(int idx) { return matches[idx]; }

    // getters

    PyTensor* py_tensor(int idx) {
        PyObject* val = args[use][idx];
        return ((PyTensor*)val);
    }
    sail::Tensor tensor(int idx) {
        PyObject* val = args[use][idx];
        return ((PyTensor*)val)->tensor;
    }
    std::vector<long> int_list(int idx) {
        PyObject* arg = args[use][idx];
        std::vector<long> ret;
        if (arg == nullptr || arg == Py_None) {
            return ret;
        }

        auto tuple = PyTuple_Check(arg);

        auto size = tuple ? PyTuple_Size(arg) : PyList_Size(arg);
        for (int i = 0; i < size; i++) {
            PyObject* val =
                tuple ? PyTuple_GetItem(arg, i) : PyList_GetItem(arg, i);
            ret.push_back(PyLong_AsLong(val));
        }
        // std::reverse(ret.begin(), ret.end());

        return ret;
    }
    std::vector<double> float_list(int idx) {
        PyObject* arg = args[use][idx];
        std::vector<double> ret;

        auto tuple = PyTuple_Check(arg);

        auto size = tuple ? PyTuple_Size(arg) : PyList_Size(arg);
        for (int i = 0; i < size; i++) {
            PyObject* val =
                tuple ? PyTuple_GetItem(arg, i) : PyList_GetItem(arg, i);
            if (PyFloat_Check(arg)) {
                ret.push_back(PyFloat_AsDouble(val));
            } else {
                ret.push_back((double)PyLong_AsLong(arg));
            }
        }
        // std::reverse(ret.begin(), ret.end());

        return ret;
    }
    std::vector<sail::Tensor> tensor_list(int idx) {
        PyObject* arg = args[use][idx];
        std::vector<sail::Tensor> ret;

        auto tuple = PyTuple_Check(arg);

        auto size = tuple ? PyTuple_Size(arg) : PyList_Size(arg);
        for (int i = 0; i < size; i++) {
            PyObject* val =
                tuple ? PyTuple_GetItem(arg, i) : PyList_GetItem(arg, i);
            ret.push_back(((PyTensor*)val)->tensor);
        }
        // std::reverse(ret.begin(), ret.end());

        return ret;
    }
    bool boolean(int idx) {
        PyObject* arg = args[use][idx];
        if (arg == nullptr || arg == Py_None) {
            return signatures[use].get_default_bool(idx);
        }
        return PyObject_IsTrue(arg);
    }
    double double_(int idx) {
        PyObject* arg = args[use][idx];
        if (arg == nullptr || arg == Py_None) {
            return signatures[use].get_default_double(idx);
        }
        if (PyFloat_Check(arg)) {
            return PyFloat_AsDouble(arg);
        }
        return (double)PyLong_AsLong(arg);
    }
    int64_t integer(int idx) {
        PyObject* arg = args[use][idx];
        if (arg == nullptr || arg == Py_None) {
            return signatures[use].get_default_int(idx);
        }
        return PyLong_AsLong(arg);
    }
    bool isNone(int idx) {
        return (args[use][idx] == Py_None || args[use][idx] == nullptr);
    }
    int64_t int_as_axis(int idx) {
        PyObject* arg = args[use][idx];
        if (arg == nullptr || arg == Py_None || arg == NULL) {
            return signatures[use].get_default_int_or_axis(idx);
        }
        return PyLong_AsLong(arg);
    }
    std::vector<long> int_list_as_axis(int idx) {
        PyObject* arg = args[use][idx];
        std::vector<long> ret;
        if (arg == nullptr || arg == Py_None) {
            return {NULLDIM};
        }

        auto tuple = PyTuple_Check(arg);

        auto size = tuple ? PyTuple_Size(arg) : PyList_Size(arg);
        for (int i = 0; i < size; i++) {
            PyObject* val =
                tuple ? PyTuple_GetItem(arg, i) : PyList_GetItem(arg, i);
            ret.push_back(PyLong_AsLong(val));
        }
        // std::reverse(ret.begin(), ret.end());

        return ret;
    }
    std::vector<long> int_as_list(int idx) {
        PyObject* arg = args[use][idx];
        if (arg == nullptr || arg == Py_None) {
            return std::vector<long>(1, signatures[use].get_default_int(idx));
        }
        return std::vector<long>(1, PyLong_AsLong(arg));
    }
    std::string string(int idx) {
        PyObject* arg = args[use][idx];
        if (arg == nullptr || arg == Py_None) {
            return signatures[use].get_default_string(idx);
        }
        return std::string(PyUnicode_AsUTF8(arg));
    }
};
// }  // namespace sail
// if (result == false) {
//     for (FunctionSignature<N> s : signatures) {
//         sig_strings += s.signature + "\n";
//     }
//     THROW_ERROR_DETAILED(
//         SailCError,
//         "Arguments do not match defined function signatures: \n",
//         sig_strings);
// }