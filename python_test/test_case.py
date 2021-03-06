import unittest
import numpy as np
import sail, random
import time, traceback
from multiprocessing.pool import ThreadPool

# import tracemalloc
# tracemalloc.start(10)



def numel(x):
    s = x.shape 
    o = 1
    for i in s:
        o *= i 
    return o


def dictionary_to_vector(dictionary):
    return np.concatenate([dictionary[i].ravel() for i in dictionary])

def vector_to_dictionary(vector, dictionary):
    nd = {}
    start = 0
    for d in dictionary:
        nd[d] = None
        data = dictionary[d]
        end = start + int(numel(data))   
        nd[d] = vector[start:end].reshape(data.shape)
        start += int(numel(data))   
    return nd

def check_gradients_vector(forward_fcn, param_dictionary, rtol=1e-5, atol=1e-8, eps=1e-3):
    params = [sail.Tensor(param_dictionary[a], requires_grad=True) for a in param_dictionary]
    output = forward_fcn(*params)
    output.backward()
    grads_dic = {}
    for i, p in enumerate(param_dictionary):
        grads_dic[p] = params[i].grad.numpy() 

    parameters = dictionary_to_vector(param_dictionary)
    grads = dictionary_to_vector(grads_dic)
    num_params = len(parameters)

    j_plus = np.zeros(num_params)
    j_minus = np.zeros(num_params)
    grad_approx = np.zeros(num_params)

    for i in range(len(parameters)):
        # pass
        params_plus = np.copy(parameters)
        params_plus[i] += eps 
        params_plus_dict = vector_to_dictionary(params_plus, param_dictionary)
        params_plus_ = [sail.Tensor(params_plus_dict[a]) for a in params_plus_dict]
        
        z = forward_fcn(*params_plus_)
        j_plus[i] = z.numpy()

        params_minus = np.copy(parameters)
        params_minus[i] -= eps 
        params_minus_dict = vector_to_dictionary(params_minus, param_dictionary)
        params_minus_ = [sail.Tensor(params_minus_dict[a]) for a in params_minus_dict]
        
        z = forward_fcn(*params_minus_)
        j_minus[i] = z.numpy()

        grad_approx[i] = (j_plus[i] - j_minus[i])/(2 * eps)

    return np.allclose(np.array(grad_approx), np.array(grads), rtol=rtol, atol=atol)


class RunCounter():

    global_runs = 0
    global_failures = 0
    global_pass = 0
    global_errors = {}
    def __init__(self):
        self.runs = 0
        self.pass_ = 0
        self.failures = 0
        
    def log_run(self, failure, fcn="", failure_message=""):
        self.runs += 1
        RunCounter.global_runs += 1
        if (failure):
            self.failures += 1
            RunCounter.global_failures += 1
            if fcn in RunCounter.global_errors:
                RunCounter.global_errors[fcn] += [failure_message]
            else:
                RunCounter.global_errors[fcn] = [failure_message]
        else:
            self.pass_ += 1
            RunCounter.global_pass += 1

def requires_grad_decorator(func):
    def wrapper(self):
        func(self, False)
        func(self, True)
    return wrapper

def dtype_decorator(func):
    args = [[sail.float64, np.float64], 
            [sail.float32, np.float32], 
            [sail.int64, np.int64], 
            [sail.uint64, np.uint64], 
            [sail.int32, np.int32], 
            [sail.uint32, np.uint32], 
            [sail.int16, np.int16], 
            [sail.uint16, np.uint16], 
            [sail.int8, np.int8], 
            [sail.uint8, np.uint8]]
    def wrapper(self):
        for i in range(len(args)):
            for j in range(i, len(args)):
                func(self, args[i], args[j])
    return wrapper

def run(c):
    c.run()

class UnitTest():

    @staticmethod
    def execute():
        e_code = 0
        classes = UnitTest.__subclasses__()
        classes = [C() for C in classes]
        # random.shuffle(classes)
        # p = ThreadPool(8)
        # xs = p.map(run, classes)
        # p.close()
        [c.run() for c in classes]

        if RunCounter.global_failures != 0:
            e_code = 1
            print ("Errors: ")
            for r in RunCounter.global_errors:
                print (r)
                print ("="*80)
                for i in RunCounter.global_errors[r]:
                    print (i)
                    print ("-"*80)
        else:
            print ("No Errors!")

        print ("Test Pass Percentage: %s%%" % (int(RunCounter.global_pass/RunCounter.global_runs * 100)))
        exit(e_code)

    def assertion(f):
        def inner(self, *args, **kwargs):
            try:
                f(self, *args, **kwargs)
            except AssertionError as e:
                self.runner.log_run(True, self.__class__.__name__ + "." + self.t, failure_message=traceback.format_exc())
                return 
            self.runner.log_run(False)
        return inner 
                

    def __init__(self):
        self.runner = RunCounter()
        self.tests = [a for a in dir(self) if a.startswith("test")]
        self.num_tests = len(self.tests)
        self.snapshots = []
        self.assertions = 0

    @assertion
    def assert_eq(self, a, b=True):
        assert a == b, (a, b)

    @assertion
    def assert_lt(self, a, b):
        assert a < b, (a, b)
    @assertion
    def assert_lte(self, a, b):
        assert a <= b, (a, b)
    @assertion
    def assert_gt(self, a, b):
        assert a > b, (a, b)
    @assertion
    def assert_gte(self, a, b):
        assert a >= b, (a, b)
    @assertion
    def assert_true(self, a):
        assert a

    @assertion
    def assert_np_array_equal(self, arr1, arr2):
        assert (np.array_equal(arr1, arr2)), (arr1, arr2)

    @assertion
    def assert_throws(self, call, args, error):
        try:
            call(*args)
        except Exception as e:
            assert(e.__class__.__name__, error().__class__.__name__)


    def assert_eq_np(self, arr1, arr2, eps=None):
        # print (np.array_equal(np_arr, sail_np))
        if (eps):
            diff = abs(arr1 - arr2)
            md = np.nanmax(diff)
            self.assert_lt(md, eps)
            return
        
        self.assert_np_array_equal(arr1, arr2)

    @assertion
    def assert_neq_np(self, arr1, arr2, eps=None):
        # print (np.array_equal(np_arr, sail_np))
        
        assert (not np.array_equal(arr1, arr2)), (arr1, arr2)

    def assert_eq_np_sail(self, np_arr, sail_arr, eps=None):
        sail_np = sail_arr.numpy()
        try:
            sail_np[sail_np == -np.inf] = 0
            sail_np[sail_np == np.inf] = 0
            sail_np[sail_np == np.nan] = 0
            sail_np[sail_np == -np.nan] = 0
            np_arr[np_arr == -np.inf] = 0
            np_arr[np_arr == np.inf] = 0
            np_arr[np_arr == np.nan] = 0
            np_arr[np_arr == -np.nan] = 0
        except:
            pass

        self.assert_eq_np(np_arr, sail_np, eps=eps)

    def run(self):
        for t in self.tests:
            self.t = t 
            getattr(self, t)()
        
        self.log_complete()


    def log_complete(self):
        print ("INFO | %s Results: %s/%s | Total: %s/%s" % (self.__class__.__name__,
                 self.runner.pass_, self.runner.runs, RunCounter.global_pass, RunCounter.global_runs))

    def tearDown(self):
        print ("ya")
    