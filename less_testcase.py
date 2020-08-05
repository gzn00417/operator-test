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


class LessTestCase(OperatorTestCase):

    def init_data(self):
        x1 = [
            [[-1, -1, -1],
             [4, 5, 6],
             [1, 1, 1]],

            [[-1, -2, -3],
             [1, 1, 1],
             [7, 8, 9]],
        ]

        x2 = [
            [[-2, -2, -2],
             [-2, -2, 2],
             [-2, -2, 2]],

            [[-1, -2, -3],
             [4, 5, 6],
             [1, 1, 1]],
        ]

        self.x = [np.array([x1]), np.array([x2])]
        self.op_name = 'less'

        super(self.__class__, self).init_data()

    def tf_net(self):
        # 输入
        inputs = self.inputs()

        # 算子
        output = tf.less(inputs[0], inputs[1], name=self.output_name)
        return output

    def test(self):
        self.init_data()
        tf_rst = self.save_ckpt()
        self.ckpt_to_pb('models/%s.ckpt' % self.op_name,
                        'models/%s.pb' % self.op_name, self.output_name)
        self.pb_to_caffe()
        # ckpt_rst = self.restore_from_ckpt()
        # self.assertEqual(np.array_equal(ckpt_rst, tf_rst), True)
        # pb_rst = self.restore_from_pb()
        # self.assertEqual(np.array_equal(pb_rst, tf_rst), True)
        caffe_rst = self.restore_from_caffe()

        print("###################################")
        print("###################################")
        print("tensorflow result = %s" % tf_rst)
        print("###################################")
        print("caffe result = %s" % caffe_rst)
        print("###################################")
        print("###################################")

        caffe_rst = np.around(caffe_rst, decimals=4)
        caffe_true_false_rst = caffe_rst > 0
        self.assertEqual(np.array_equal(caffe_true_false_rst, tf_rst), True)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(LessTestCase("test"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
