# -*- coding: utf-8 -*-)
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.initialize_all_variables())  #比较旧一点的初始化变量方法
    print w1
    print sess.run(w1)
