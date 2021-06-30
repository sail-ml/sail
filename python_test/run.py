from test_case import *
import numpy as np
import sail
import time
import unittest
import faulthandler

import basic_ops_test
import loss_test
import shape_test
import linalg_test
import layer_test
import reduction_test
import factory_test 
import cat_test 
import cast_test 
import print_test

import integration_tests.basic_mlp
import integration_tests.integration_test1

if __name__ == "__main__":
    faulthandler.enable()
    UnitTest.execute()
