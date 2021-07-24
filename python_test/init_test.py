from test_case import *
import numpy as np
import sail
import time
from scipy.stats import normaltest, ttest_1samp
import unittest, random

shapes = [(3200, 10000)]

alpha = 0.05
gain = np.sqrt(2)
def calculate_fan_in_out(x):

    num_input_fmaps = x.shape[1]
    num_output_fmaps = x.shape[0]
    receptive_field_size = 1
    if (len(x.shape) > 2):
        for i in range(2, len(x.shape)):
            receptive_field_size *= x.shape[i]
    
    fan_in = num_input_fmaps * receptive_field_size;
    fan_out = num_output_fmaps * receptive_field_size;
    return (fan_in, fan_out)#std::make_tuple(fan_in, fan_out);

class UtilsTest(UnitTest):

    def test_nonlin_not_supported(self):
        
        for sh in shapes:
            x = sail.random.uniform(0, 1, sh)
            self.assert_throws(sail.init.kaiming_normal, (x, "fan_in", "reluz"), sail.SailError)

        return

    def test_shape_too_small(self):
        
        x = sail.random.uniform(0, 1, (12))
        self.assert_throws(sail.init.kaiming_normal, (x, "fan_in", "relu"), sail.SailError)

        return

class XavierUniform(UnitTest):

    def test_base(self):
        
        for sh in shapes:
            x = sail.random.uniform(0, 1, sh)
            sail.init.xavier_uniform(x, gain=0.5)

            fan_in, fan_out = calculate_fan_in_out(x)

            y = x.numpy()
            self.assert_lte(np.max(y), 0.5 * np.sqrt(6 / (fan_in + fan_out)))
            self.assert_gte(np.min(y), 0.5 * -np.sqrt(6 / (fan_in + fan_out)))
         
        return

class XavierNormal(UnitTest):

    def test_base(self):
        
        for sh in shapes:
            x = sail.random.normal(0, 1, sh)
            sail.init.xavier_normal(x, gain=0.5)

            y = x.numpy()

            _, p_val = normaltest(y,axis=None)

            # self.assert_lte(np.max(y), 0.5 * np.sqrt(2 / (fan_in + fan_out)))
            self.assert_gte(p_val, alpha)

        return

class KaimingUniform(UnitTest):

    def test_base(self):
        
        for sh in shapes:
            x = sail.random.uniform(0, 1, sh)
            sail.init.kaiming_uniform(x, mode="fan_in", nonlin="relu")

            fan_in, fan_out = calculate_fan_in_out(x)

            std = gain / np.sqrt(fan_in)
            bound = np.sqrt(3) * std + 1e-5

            y = x.numpy()
            self.assert_lte(np.max(y), bound)
            self.assert_gte(np.min(y), -bound)
         
        return

    def test_fan_out(self):
        
        for sh in shapes:
            x = sail.random.uniform(0, 1, sh)
            sail.init.kaiming_uniform(x, mode="fan_out", nonlin="relu")

            fan_in, fan_out = calculate_fan_in_out(x)

            std = gain / np.sqrt(fan_out)
            bound = np.sqrt(3) * std + 1e-5

            y = x.numpy()
            self.assert_lte(np.max(y), bound)
            self.assert_gte(np.min(y), -bound)
         
        return

    def test_mode(self):
        
        for sh in shapes:
            x = sail.random.normal(0, 1, sh)
            self.assert_throws(sail.init.kaiming_uniform, (x, "fnas", "relu"), sail.SailError)

        return

class KaimingNormal(UnitTest):

    def test_base(self):
        
        for sh in shapes:
            x = sail.random.normal(0, 1, sh)
            sail.init.kaiming_normal(x, mode="fan_in", nonlin="relu")

            y = x.numpy()

            _, p_val = normaltest(y,axis=None)


            # self.assert_lte(np.max(y), 0.5 * np.sqrt(2 / (fan_in + fan_out)))
            self.assert_gte(p_val, alpha)
         
        return

    def test_fan_out(self):
        
        for sh in shapes:
            x = sail.random.normal(0, 1, sh)
            sail.init.kaiming_normal(x, mode="fan_out", nonlin="relu")

            y = x.numpy()

            _, p_val = normaltest(y,axis=None)


            # self.assert_lte(np.max(y), 0.5 * np.sqrt(2 / (fan_in + fan_out)))
            self.assert_gte(p_val, alpha)
         
        return

    def test_mode(self):
        
        for sh in shapes:
            x = sail.random.normal(0, 1, sh)
            self.assert_throws(sail.init.kaiming_normal, (x, "fnas", "relu"), sail.SailError)

        return
