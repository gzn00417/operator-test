#!/usr/bin/env python
# coding=utf-8
"""module docstring"""

import tensorflow as tf
import numpy as np
import unittest

per_process_gpu_memory_fraction = 0.01

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
        tf.Variable(tf.zeros([1]), name='nouse')

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
        '''
        保存模型
        '''
        self.tf_net()
        saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型
        var_init = tf.global_variables_initializer()

        # input and output
        feed_dict = dict()

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

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) \
                as sess:
            sess.run(var_init)
            result = sess.run(output, feed_dict=feed_dict)
            saver.save(sess, "models/%s.ckpt" % self.op_name)
            return result

    def restore_weight_from_ckpt(self):
        '''
        只从ckpt读取权重
        '''
        tf.reset_default_graph()

        self.tf_net()
        saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型

        feed_dict = dict()
        # input and output
        for i in range(0, len(self.x)):
            input_name = "%s_%s" % (self.input_name, i)
            input = tf.get_default_graph().get_tensor_by_name(
                "%s:0" % input_name)
            feed_dict[input] = self.x[i]

        output = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % self.output_name)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) \
                as sess:
            saver.restore(sess, 'models/%s.ckpt' % self.op_name)
            return sess.run(output, feed_dict=feed_dict)

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

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) \
                as sess:
            saver.restore(sess, 'models/%s.ckpt' % self.op_name)
            return sess.run(output, feed_dict=feed_dict)

    def ckpt_to_pb(self, ckpt_path, pb_path, output_node_names):
        '''
        通用方法，ckpt转pb
        :param ckpt_path: xxx.ckpt(千万不要加后面的xxx.ckpt.data这种，到ckpt就行了!)
        :param output_graph: PB模型保存路径
        :param output_node_names: 模型输出节点名称，该节点名称必须是原模型中存在的节点
        :return:
        '''
        from tensorflow import graph_util
        saver = tf.train.import_meta_graph(ckpt_path + '.meta',
                                           clear_devices=True)
        graph = tf.get_default_graph()  # 获得默认的图

        input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) \
                as sess:
            saver.restore(sess, ckpt_path)  # 恢复图并得到数据
            output_graph_def = graph_util.convert_variables_to_constants(
                # 模型持久化，将变量值固定
                sess=sess,
                input_graph_def=input_graph_def,  # 等于:sess.graph_def
                output_node_names=output_node_names.split(
                    ","))  # 如果有多个输出节点，以逗号隔开

            with tf.gfile.GFile(pb_path, "wb") as f:  # 保存模型
                f.write(output_graph_def.SerializeToString())  # 序列化输出

    def restore_from_pb(self):
        '''
        测试pb
        '''
        tf.reset_default_graph()

        pb_path = 'models/%s.pb' % self.op_name
        with open(pb_path, "rb") as f:
            output_graph_def = tf.GraphDef()
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        feed_dict = dict()
        # input and output
        for i in range(0, len(self.x)):
            input_name = "%s_%s" % (self.input_name, i)
            input = tf.get_default_graph().get_tensor_by_name(
                "%s:0" % input_name)
            feed_dict[input] = self.x[i]
        output = tf.get_default_graph().get_tensor_by_name(
            "%s:0" % self.output_name)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) \
                as sess:
            return sess.run(output, feed_dict=feed_dict)
        ##############################################################
        ##############################################################

        print("###################################")
        print("###################################")
        print("tensorflow result = %s" % tf_rst)
        print("###################################")
        print("###################################")
        print("###################################")

        tf_rst = np.around(tf_rst, decimals=3)
        self.assertEqual(np.array_equal(caffe_rst, tf_rst), True)
