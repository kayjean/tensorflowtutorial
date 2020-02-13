import tensorflow as tf
import numpy as np


""" 2-D """
print("2-D:")
sess = tf.Session()
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])  # 2-D tensor `a`
print("a = ", sess.run(a))
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])  # 2-D tensor `b`
print("b = ", sess.run(b))
c = tf.matmul(a, b)
print("c = ", sess.run(c))
""" 3-D """
print("3-D:")
a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])  # 3-D tensor `a`
print("a = ", sess.run(a))
b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])  # 3-D tensor `b`
print("b = ", sess.run(b))
c = tf.matmul(a, b)
print("c = ", sess.run(c))
sess.close()



x=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])  
y=tf.constant([[0,0,1.0],[0,0,1.0],[0,0,1.0]])
z=tf.multiply(x,y)

x1=tf.constant(1)
y1=tf.constant(2)
z1=tf.multiply(x1,y1)

x2=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
y2=tf.constant(2.0)
z2=tf.multiply(x2,y2)

x3=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])  
y3=tf.constant([[0,0,1.0],[0,0,1.0],[0,0,1.0]])
z3=tf.matmul(x3,y3)

with tf.Session() as sess:
    print("example1")
    print(sess.run(z))
    print("example2")
    print(sess.run(z1))
    print("example3")
    print(sess.run(z2))
    print("example4")
    print(sess.run(z3))