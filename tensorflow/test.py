import numpy as np
import tensorflow as tf

session = tf.Session()

x_vas = np.float32(np.array([1., 2., 3., 4.]))
with tf.name_scope('x_data'):
    x_data = tf.placeholder(tf.float32)
    tf.summary.histogram('x_data', x_data)
with tf.name_scope('const'):
    m_const = tf.constant(3.)
    tf.summary.scalar('const', m_const)
with tf.name_scope('result'):
    my_product = tf.multiply(x_data, m_const)
    tf.summary.scalar('result', my_product)

merge_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('output/', session.graph)

if __name__ == '__main__':
    i = 0
    for x_val in x_vas:
        i = i + 1
        session.run(my_product, feed_dict={x_data: x_val})
        result = session.run(merge_summary, feed_dict={x_data: x_val})
        writer.add_summary(result)
