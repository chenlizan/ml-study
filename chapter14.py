import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

log_dir = ".logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

g = tf.Graph()

with g.as_default():
    t1 = tf.constant(np.pi)
    t2 = tf.constant([1, 2, 3, 4])
    t3 = tf.constant([[1, 2], [3, 4]])

    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)

    print(r1)
    print(r2)
    print(r3)

    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()

    print("Shapes:", s1, s2, s3)

    with tf.compat.v1.Session(graph=g):
        print("Ranks:", r1.eval(), r2.eval(), r3.eval())

with g.as_default():
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    c = tf.constant(3, name='c')

    z = 2 * (a - b) + c

with tf.compat.v1.Session(graph=g) as sess:
    print('2 * (a - b) + c =>', sess.run(z))

with g.as_default():
    tf_a = tf.compat.v1.placeholder(tf.int32, shape=[], name='tf_a')
    tf_b = tf.compat.v1.placeholder(tf.int32, shape=[], name='tf_b')
    tf_c = tf.compat.v1.placeholder(tf.int32, shape=[], name='tf_c')

    r1 = tf_a - tf_b
    r2 = 2 * r1
    z = r2 + tf_c

    with tf.compat.v1.Session(graph=g) as sess:
        feed = {tf_a: 1, tf_b: 2, tf_c: 3}
        print('z:', sess.run(z, feed_dict=feed))

with g.as_default():
    tf_x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name='tf_x')
    x_mean = tf.reduce_mean(tf_x, axis=0, name='mean')

np.random.seed(123)
np.set_printoptions(precision=2)
with tf.compat.v1.Session(graph=g) as sess:
    x1 = np.random.uniform(low=0, high=1, size=(5, 2))
    print("Feeding data with shape ", x1.shape)
    print("Result:", sess.run(x_mean, feed_dict={tf_x: x1}))

    x2 = np.random.uniform(low=0, high=1, size=(10, 2))
    print("Feeding data with shape ", x2.shape)
    print("Result:", sess.run(x_mean, feed_dict={tf_x: x2}))

with g.as_default():
    w = tf.Variable(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), name="w")
    print(w)

with tf.compat.v1.Session(graph=g) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(w))

with g.as_default():
    w1 = tf.Variable(1, name="w1")
    init_op = tf.compat.v1.global_variables_initializer()
    w2 = tf.Variable(2, name="w2")

with tf.compat.v1.Session(graph=g) as sess:
    sess.run(init_op)
    # print(sess.run(w2))

with g.as_default():
    with tf.compat.v1.variable_scope("net_A"):
        with tf.compat.v1.variable_scope("layer-1"):
            w1 = tf.Variable(tf.compat.v1.random_normal(shape=(10, 4)), name="weights")
        with tf.compat.v1.variable_scope("layer-2"):
            w2 = tf.Variable(tf.compat.v1.random_normal(shape=(20, 10)), name="weights")
    with tf.compat.v1.variable_scope("net_B"):
        with tf.compat.v1.variable_scope("layer-1"):
            w3 = tf.Variable(tf.compat.v1.random_normal(shape=(10, 4)), name="weights")
print(w1)
print(w2)
print(w3)


def build_classifier(data, labels, num_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.compat.v1.get_variable(name="weights", shape=[data_shape[1], num_classes], dtype=tf.float32)
    bias = tf.compat.v1.get_variable(name="bias", initializer=tf.zeros(shape=num_classes))
    logits = tf.add(tf.matmul(data, weights), bias, name="logits")
    print(logits)
    return logits, tf.nn.softmax(logits)


def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(tf.compat.v1.random_normal(shape=(data_shape[1], n_hidden)), name="w1")
    b1 = tf.Variable(tf.compat.v1.zeros(shape=n_hidden), name="b1")
    hidden = tf.add(tf.matmul(data, w1), b1, name="hidden_pre-activation")
    hidden = tf.nn.relu(hidden, "hidden_activation")
    w2 = tf.Variable(tf.compat.v1.random_normal(shape=(n_hidden, data_shape[1])), name="w2")
    b2 = tf.Variable(tf.compat.v1.zeros(shape=data_shape[1]), name="b2")
    output = tf.add(tf.matmul(hidden, w2), b2, name="output")
    return output, tf.nn.sigmoid(output)


batch_size = 64

with g.as_default():
    tf_X = tf.compat.v1.placeholder(shape=[batch_size, 2], dtype=tf.float32, name='tf_X')

    with tf.compat.v1.variable_scope("generator"):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)

    with tf.compat.v1.variable_scope("classifier") as scope:
        cls_out1 = build_classifier(data=tf_X, labels=tf.ones(shape=batch_size))
        scope.reuse_variables()
        cls_out2 = build_classifier(data=cls_out1[1], labels=tf.zeros(shape=batch_size))
        print(cls_out2)

        init_op = tf.compat.v1.global_variables_initializer()

with g.as_default():
    tf.compat.v1.set_random_seed(123)

    tf_x = tf.compat.v1.placeholder(shape=None, dtype=tf.float32, name='tf_x')
    tf_y = tf.compat.v1.placeholder(shape=None, dtype=tf.float32, name='tf_y')
    weight = tf.Variable(tf.compat.v1.random_normal(shape=(1, 1), stddev=0.25), name="weight")
    bias = tf.Variable(0.0, name="bias")
    y_hat = tf.add(weight * tf_x, bias, name="y_hat")
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), name="cost")
    optim = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name="train_op")

np.random.seed(0)


def make_random_data():
    x = np.random.uniform(low=2, high=4, size=100)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.5 + t * t / 3), size=None)
        y.append(r)
    return x, 1.726 * x - 0.84 + np.array(y)


x, y = make_random_data()

plt.plot(x, y, 'o')
plt.show()

x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]

n_epochs = 500
training_costs = []
with tf.compat.v1.Session(graph=g) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for e in range(n_epochs):
        # c, _ = sess.run([cost, train_op], feed_dict={tf_x: x_train, tf_y: y_train})

        c, _ = sess.run(['cost:0', 'train_op'], feed_dict={'tf_x:0': x_train, 'tf_y:0': y_train})

        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, c))

plt.plot(training_costs)
plt.show()
