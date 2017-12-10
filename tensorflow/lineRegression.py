import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class lineRegression(object):
    def __init__(self, input_num):
        self.input_num = input_num
        self.weights = tf.constant(0, shape=[1, input_num + 1])

    def get_output(self, input_data, labels, step_rate):
        if self.is_inverse(input_data):
            return
        else:
            return self.gradient_decend(input_data, labels, step_rate)

    def is_inverse(self, input_data):
        return tf.matrix_inverse(tf.matmul(input_data, input_data, False, True))

    def gradient_decend(self, input_data, labels):
        pass
