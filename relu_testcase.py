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


class ReluTestCase(OperatorTestCase):

    def init_data(self):
        x1 = [
            [[-1, -1, -1],
             [4, 5, 6],
             [1, 1, 1]],

            [[-1, -2, -3],
             [1, 1, 1],
             [7, 8, 9]],
        ]

        self.x = [np.array([x1])]
        self.op_name = 'relu'

        super(self.__class__, self).init_data()

    def tf_net(self):
        # 输入
        inputs = self.inputs()

        # 算子
        output = tf.nn.relu(inputs[0], name=self.output_name)
        return output


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(ReluTestCase("test"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
