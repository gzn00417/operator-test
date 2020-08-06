#!/usr/bin/env python
# coding=utf-8
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
             [1, 1, 1]]
        ]
        self.x = [np.array([x1]), np.array([x2])]
        #self.op_name = 'Neg' 可加可不加
        #将父类对象转换为子类
        super(self.__class__, self).init_data()
        #定义网络
    def tf_net(self):
        inputs = self.inputs()
        #inputs可以读取多个数据输入 如果仅测试一个数据（比如x1）则改为input[0]
        output = tf.negative(inputs,name=self.output_name)
        return output

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(NegTestCase("test"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
