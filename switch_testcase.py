#!/usr/bin/env python
# coding=utf-8
"""module docstring"""


import numpy as np
import unittest
import os
import sys
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops as cf


this_dir = os.getcwd()
sys.path.append(os.path.join(this_dir, '..'))

from operator_testcase import OperatorTestCase


class SwitchTestCase(OperatorTestCase):

    def init_data(self):
        x1 = 3
        x2 = 2

        self.x = [np.array([x1]), np.array([x2])]
        self.op_name = 'switch'
        super(self.__class__, self).init_data()

    def tf_net(self):
        # 输入
        inputs = self.inputs()
        # 算子
        pred = tf.less(inputs[0], inputs[1], name='pred')
        outputs = cf.switch(inputs[0], pred[0], name=self.output_name)

    def test(self):
        self.init_data()
        tf_rst = self.save_ckpt()
        self.ckpt_to_pb('models/%s.ckpt' % self.op_name,
                        'models/%s.pb' % self.op_name, self.output_name)
        # # pb_rst = self.restore_from_pb(); print(pb_rst)
        self.pb_to_caffe()
        # self.tf_net()
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

        tf_rst = np.around(tf_rst, decimals=4)
        caffe_rst = np.around(caffe_rst, decimals=4)

        self.assertEqual(np.array_equal(caffe_rst, tf_rst), True)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(SwitchTestCase("test"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
