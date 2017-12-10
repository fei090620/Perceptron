import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf

session = tf.Session()
iris = datasets.load_iris()
binary_target = np.array([1. if x == 0 else 0. for x in iris.target])

iris_2d = np.array([[x[2], x[3]] for x in iris.data])

x1_data = tf.placeholder(tf.float32, shape=[None, 1])
x2_data = tf.placeholder(tf.float32, shape=[None, 1])

y_target = tf.placeholder(tf.float32, shape=[None, 1])

A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

my_mult = tf.multiply(x1_data, A)
my_add = tf.add(my_mult, b)
my_sub = tf.subscribe(x2_data, my_add)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(my_sub, y_target)

my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

init = tf.initialize_all_variables()
session.run(init)





if __name__ == '__main__':
    print iris.target, iris.data
