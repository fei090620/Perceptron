import tensorflow as tf

x_data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
          [0, 0, 0, 0], [1, 1, 1, 1],
          [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
          [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1],
          [1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1]]

y = [1, 1, 1, 1,
     0, 1, 1, 1, 1,
     0, 0, 0, 1, 1, 0, 0]

session = tf.Session()

with tf.name_scope('Perceptron'):
    with tf.name_scope('input'):
        x_input = tf.Variable(tf.random_uniform([16, 4], 0, 1))
        tf.summary.tensor_summary('input', x_input)

    with tf.name_scope('Weights'):
        W = tf.Variable(tf.zeros([1, 4]))
        tf.summary.tensor_summary('Weights', W)

    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros([1]))
        tf.summary.histogram('bias', b)

    with tf.name_scope('pre_output'):
        pre_y = tf.add(tf.matmul(W, x_input, False, True), b)
        loss = tf.reduce_mean(tf.square(y - pre_y))
        tf.summary.scalar('pre_output', loss)

optimizer = tf.train.GradientDescentOptimizer(0.1)
one_step_train = optimizer.minimize(loss)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('output/', session.graph)

init = tf.global_variables_initializer()

session.run(init)

if __name__ == '__main__':
    for i in xrange(0, 500):
        session.run(one_step_train)
        if i % 10 == 0:
            result = session.run(merged)
            writer.add_summary(result, i)
            print i, session.run(W), session.run(b)
