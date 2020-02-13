import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

g1 = tf.Graph()
with g1.as_default():
    x = tf.constant(1)
    y = tf.constant(1)
    sol = tf.add(x,y) # add x and y

with tf.Session(graph=g1) as sess: 
    print(sol) # print tensor, not their value

with tf.Session(graph=g1) as sess: 
    print(sol.eval()) # evaluate their value

s1 = tf.Session(graph=g1)
print(s1.run(sol)) # another way of evaluating value

g2 = tf.Graph()
with g2.as_default():
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)
    sol = tf.add(x,y) # add x and y

s2 = tf.Session(graph=g2)

print(s2.run(sol, feed_dict={x: 2, y: 3})) 
print(s2.run(sol, feed_dict={x: 5, y: 7})) 