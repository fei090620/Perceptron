import tensorflow as tf
import matplotlib.pyplot as plt

session = tf.Session()

x_vals = tf.lin_space(-1., 1., 500)
target = tf.constant(0.)

#l1 l2
l2_y_vals = tf.square(target - x_vals)
l1_y_vals = tf.abs(target - x_vals)

#pseudo-huber
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)

delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2)) - 1.)


if __name__ == '__main__':
    _x, l1_y_out, l2_y_out, phuber1_y_out, phuber2_y_out = session.run([x_vals, l1_y_vals, l2_y_vals, phuber1_y_vals, phuber2_y_vals])
    plt.plot(_x, l1_y_out, label='l1')
    plt.plot(_x, l2_y_out, label='l2')
    plt.plot(_x, phuber1_y_out, label='phuber_025')
    plt.plot(_x, phuber2_y_out, label='phuber_5')

    plt.legend()
    plt.show()