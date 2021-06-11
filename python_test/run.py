from test_case import *
import numpy as np
import sail
import time
import unittest
import faulthandler

import basic_ops_test, shape_test, linalg_test, layer_test, reduction_test

# import resource, tracemalloc
# from pympler.tracker import SummaryTracker

# tracemalloc.start(100)

  
# def limit_memory(maxsize):
#     soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#     resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

if __name__ == "__main__":
    # limit_memory(4e+9)
    faulthandler.enable()
    # t0 = tracemalloc.take_snapshot()
    # tracker = SummaryTracker()
    UnitTest.execute()
    # tracker.print_diff()

    # t1 = tracemalloc.take_snapshot()
    # stats = t1.compare_to(t0, 'filename')    
    # for stat in stats[:100]:                
    #     print("{} new KiB {} total KiB {} new {} total memory blocks: ".format(stat.size_diff/1024, stat.size / 1024, stat.count_diff ,stat.count))                

    # x = sail.random.uniform(0, 1, (10,10,10,10,10))
