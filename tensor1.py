#python35 tensorflow==1.1

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'


import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

state = tf.Variable(0, name="counter")

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])


product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run([product, output], feed_dict={input1:[7.], input2:[2.]})
    print('result', result)

    b = tf.nn.dropout(a, 0.5, noise_shape = [1, 4])
    print('b noise_shape [1, 4]', sess.run(b))

    b = tf.nn.dropout(a, 0.5, noise_shape = [1, 1])
    print('b noise_shape [1, 1]', sess.run(b))

