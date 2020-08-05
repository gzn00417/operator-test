#!/usr/bin/env python
# coding=utf-8
"""module docstring"""

import tensorflow as tf
import numpy as np
import unittest
import os
import sys
this_dir = os.getcwd()
sys.path.append(os.path.join(this_dir, '..'))

from operator_testcase import OperatorTestCase
class NegTestCase(OperatorTestCase): 
#调用OperatorTestCase类
    def init_data(self):
        data1 = [
            [[-1, -1, -1],
             [4, 5, 6],
             [1, 1, 1]],

            [[-1, -2, -3],
             [1, 1, 1],
             [7, 8, 9]],
        ]
        x1 = np.array([data1])
        self.x = [x1]
        self.op_name = 'neg'
        #将init_data对象转化为NegTestCase对象
        super(self.__class__, self).init_data()

    def tf_net(self):
        # 输入
        inputs = self.inputs()

        # 算子
        output = tf.negative(inputs[0], name=self.output_name)
        return output


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(NegTestCase("test"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
