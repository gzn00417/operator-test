#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
import unittest

#per_process_gpu_memory_fraction = 0.01
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
        //防止没有算子读进来报错 ‘加个名字nouse’
        tf.Variable(tf.zeros([1]), name='nouse')

        inputs = list()
        for i in range(len(self.x)):
            //查看TF官方文档 placeholder使用方法
            input = tf.placeholder(dtype=tf.float32, shape=[None] + self.inputshapes[i],
                                   name="%s_%s" % (self.input_name, i))
            inputs.append(input)
        return inputs

    def tf_net(self):
        pass

    def save_ckpt(self):
        '''
        保存模型
        '''
        self.tf_net()
        saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型
        var_init = tf.global_variables_initializer()

        # input and output
        feed_dict = dict()
        //保存算子输出结果
        for i in range(0, len(self.x)):
            input = tf.get_default_graph().get_tensor_by_name(
                "%s:0" % self.input_names[i])
            feed_dict[input] = self.x[i]

        default_graph = tf.get_default_graph()

        print("################ op name and type ################")
        for key, value in default_graph._nodes_by_name.items():
            print("%s: %s" % (key, value.type))
        print("################ ################ ################")

        output = default_graph.get_tensor_by_name("%s:0" % self.output_name)
       #gpu_options = tf.GPUOptions(
            #per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)

        #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) \
                #as sess:
            #sess.run(var_init)
            #result = sess.run(output, feed_dict=feed_dict)
            #saver.save(sess, "models/%s.ckpt" % self.op_name)
            #return result
    '''
    def restore_weight_from_ckpt(self):
        '''
        只从ckpt读取权重
        '''
        tf.reset_default_graph()
        self.tf_net()
        saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型
     '''
        feed_dict = dict()
        # input and output
        for i in range(0, len(self.x)):
            input_name = "%s_%s" % (self.input_name, i)
            input = tf.get_default_graph().get_tensor_by_name(
                "%s:0" % input_name)
            feed_dict[input] = self.x[i]

        output = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % self.output_name)

        ''' #这段不用看
        gpu_options = tf.GPUOptions(
           per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) \
                as sess:
            saver.restore(sess, 'models/%s.ckpt' % self.op_name)
            return sess.run(output, feed_dict=feed_dict)
            '''
        
    def restore_from_ckpt(self):
        '''
        从ckpt读取网络结构和权重
        '''
        tf.reset_default_graph()

        saver = tf.train.import_meta_graph(
            'models/%s.ckpt.meta' % self.op_name)

        feed_dict = dict()
        # input and output
        for i in range(0, len(self.x)):
            input_name = "%s_%s" % (self.input_name, i)
            input = tf.get_default_graph().get_tensor_by_name(
                "%s:0" % input_name)
            feed_dict[input] = self.x[i]

        output = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % self.output_name)
        
        from tensorflow import graph_util
        saver = tf.train.import_meta_graph(ckpt_path + '.meta',
                                           clear_devices=True)
        graph = tf.get_default_graph()  # 获得默认的图
        input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
        '''
        with tf.Session() \
                as sess:
            return sess.run(output, feed_dict=feed_dict)
         '''
        ##############################################################
        ##############################################################

        print("###################################")
        print("###################################")
        print("tensorflow result = %s" % tf_rst)
        print("###################################")

        tf_rst = np.around(tf_rst, decimals=3)

