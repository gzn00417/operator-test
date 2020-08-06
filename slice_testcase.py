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

class SliceTestCase(OperatorTestCase):

    def init_data(self):
        x1 = [
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]
        ]

        self.x = [np.array([x1])]
        #self.op_name = 'slice'

        super(self.__class__, self).init_data()

    def tf_net(self):
        # 输入
        inputs = self.inputs()

        # 算子
        output = tf.slice(inputs[0], [0, 1, 0, 0], [1, 1, 1, 3],
                          name=self.output_name)
        return output


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(SliceTestCase("test"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
