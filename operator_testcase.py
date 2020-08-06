#!/usr/bin/env python
# coding=utf-8
"""module docstring"""

import tensorflow as tf
import numpy as np
import unittest

class OperatorTestCase(unittest.TestCase):

    def init_data(self):
        self.input_name = 'input'
        self.output_name = 'output'
        self.inputshapes = list()
        self.input_names = list()

        for i in range(len(self.x)):
            inputshape = list(self.x[i].shape)
            inputshape = inputshape[1:]
            self.inputshapes.append(inputshape)

            input_name = "%s_%s" % (self.input_name, i)
            self.input_names.append(input_name)
        print("inputshapes=%s" % self.inputshapes)

    def inputs(self):
        tf.reset_default_graph()
        tf.Variable(tf.zeros([1]), name='nouse') #防止无输出导致模型报错，可以注释掉再测试
        inputs = list()
        for i in range(len(self.x)):
            input = tf.placeholder(dtype=tf.float32, shape=[None] + self.inputshapes[i],
                                   name="%s_%s" % (self.input_name, i))
            inputs.append(input)
        return inputs

    def tf_net(self):
        # implement by subclass
        pass

    def save_ckpt(self):
        #保存模型
        self.tf_net()
        #初始化参数
        var_init = tf.global_variables_initializer()
        feed_dict = dict() # input and output

        for i in range(0, len(self.x)):
            input = tf.get_default_graph().get_tensor_by_name(
                "%s:0" % self.input_names[i])
            feed_dict[input] = self.x[i]

        default_graph = tf.get_default_graph()
        output = default_graph.get_tensor_by_name("%s:0" % self.output_name)

        #启动计算图，读入数据，生成output(对应def_tf-net的output)
        with tf.Session() as sess:
            sess.run(var_init)
            result = sess.run(output, feed_dict=feed_dict)
            return result
            #需要将result写入一个txt文档保存

    def test(self):
        self.init_data()
        tf_rst = self.save_ckpt()
        #精确小数点3位 用于做输出数据精确度比对
        tf_rst = np.around(tf_rst, decimals=3)
        # caffe_rst = np.around(caffe_rst, decimals=3)
        #读入C++保存的文件 然后对比参数 是否精确到小数点3位 如果比对误差小于小数点后三位 则返回true
        #需要修改equal(caffe_rst, tf_rst) 括号中的部分
        '''
        self.assertEqual(np.array_equal(caffe_rst, tf_rst), True)
        '''
        
        
