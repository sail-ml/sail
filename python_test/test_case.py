import unittest
import numpy as np
import sail
import time, traceback
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


dic = {"a": np.random.uniform(1, 2, (35)), 
       "b": np.random.uniform(1, 2, (35)),
       }

def forward(a, b):
    c = sail.multiply(a, b)
    d = sail.sum(c)
    return d


def check_gradients_vector(forward_fcn, param_dictionary, eps=1e-3):
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
        # grad_approx[i] = grad_approx[i]#to_significant(grad_approx[i], significant=7)


    num = np.linalg.norm(grad_approx - grads)
    denom = np.linalg.norm(grad_approx) + np.linalg.norm(grads)
    diff = num/denom
    return diff 


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

class UnitTest():

    @staticmethod
    def execute():
        e_code = 0
        for C in UnitTest.__subclasses__():
            c = C()
            c.run()
        
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

        print ("Test Pass Percentage: %s%%" % (int(RunCounter.global_pass/RunCounter.global_runs) * 100))
        exit(e_code)

    def __init__(self):
        self.runner = RunCounter()
        self.tests = [a for a in dir(self) if a.startswith("test")]
        self.num_tests = len(self.tests)
        self.snapshots = []

    def assert_eq(self, a, b=True):
        try:
            assert a == b, (a, b)
        except AssertionError as e:
            self.runner.log_run(True, self.__class__.__name__, failure_message=traceback.format_exc())
            return 
        self.runner.log_run(False)

    def assert_eq_np_sail(self, np_arr, sail_arr):
        sail_np = sail_arr.numpy()
        # print (np.array_equal(np_arr, sail_np))
        self.assert_eq(np.array_equal(np_arr, sail_np))

    def run(self):
        for t in self.tests:
            # self.snapshots.append(tracemalloc.take_snapshot())
            # try:
            getattr(self, t)()
            # except AssertionError as e:
            #     self.runner.log_run(True, t, failure_message=traceback.format_exc())
            #     continue
            
            # self.runner.log_run(False)
        # stats = self.snapshots[-1].compare_to(self.snapshots[-2], 'filename')    

        # for stat in stats[:10]:                
            # print("{} new KiB {} total KiB {} new {} total memory blocks: ".format(stat.size_diff/1024, stat.size / 1024, stat.count_diff ,stat.count))                
            # for line in stat.traceback.format():                    
            #     print(line)
        
        self.log_complete()


    def log_complete(self):
        print ("INFO | %s Results: %s/%s | Total: %s/%s" % (self.__class__.__name__,
                 self.runner.pass_, self.runner.runs, RunCounter.global_pass, RunCounter.global_runs))

    def tearDown(self):
        print ("ya")
    