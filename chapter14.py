import datetime
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
    print(sess.run(w2))
