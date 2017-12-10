#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

session = tf.Session()

shap = [1, 4, 4, 1]

x_val = np.random.uniform(size=[1, 4, 4, 1])

with tf.name_scope('input'):
    x_data = tf.placeholder(tf.float32, shape=shap)
    tf.summary.histogram('input', x_data)

# 这里有需要注意的地方，卷集核的结构和输入tensor的结构定义是不同的
'''
input 的结构就是[深度，高度，宽度，信道数]
filter 的结构是[高度，宽度，输入信道数，输出信道数]
strides 的结构是 各个维度的步长

'''
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]

with tf.name_scope('con'):
    mov_avg_layers = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_avg_widnow')
    tf.summary.histogram('con', mov_avg_layers)


def define_custorm_layer(input_matrix):
    input_matrix_sequess = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    return tf.sigmoid(tf.add(tf.matmul(A, input_matrix_sequess), b))


with tf.name_scope('custorm_layer'):
    custorm_layer = define_custorm_layer(mov_avg_layers)
    tf.summary.histogram('custorm_layer', custorm_layer)

merge_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('output/', session.graph)

if __name__ == '__main__':
    for i in range(100):
        result = session.run(merge_summary, feed_dict={x_data: x_val})
        writer.add_summary(result)