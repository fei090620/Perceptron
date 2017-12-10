import tensorflow as tf
import mnist_demo.mnist_data as mnist_data

session = tf.Session()

filter_shape = [4, 4, 1, 1]

# read data
test_image, test_label, train_image, train_label = mnist_data.mnist_data_reader.read_train_test_datas()

with tf.name_scope('train_input'):
    input_X = tf.placeholder(tf.float32, shape=[1, 28, 28, 1])
    tf.summary.histogram('train_input', input_X)

with tf.name_scope('train_filter'):
    filter1 = tf.Variable(tf.random_normal(shape=filter_shape))
    tf.summary.histogram('train_filter', filter1)

with tf.name_scope('con'):
    strides = tf.constant([1, 2, 2, 1])
    conn_layer = tf.nn.conv2d(input_X, filter1, strides)
    tf.summary.histogram('conn', conn_layer)

with tf.name_scope('nn'):
    nn_input = tf.squeeze(conn_layer)
    tf.summary.histogram('nn_input', nn_input)
    with tf.name_scope('nn_hiden_1_layer'):
        with tf.name_scope('weights'):
            weights = tf.constant(0.1, shape=[])

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('output/', session.graph)

if __name__ == '__main__':
    for input_data in train_image:
        session.run(input_X, feed_dict={input_X: input_data})
        result = session.run(merged_summary, feed_dict={input_X: input_data})
        writer.add_summary(result)
