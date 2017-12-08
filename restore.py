#!/usr/bin/python


import tensorflow as tf

log_dir = '/root/tensorflow/train' 


w_conv1=tf.Variable(tf.zeros([5,5,1,32]),name='w_conv1')
b_conv1 = tf.Variable(tf.zeros([32]),name='b_conv1')
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,log_dir+"/model.ckpt")
    print("W : " , sess.run(w_conv1))
    print("b : " , sess.run(b_conv1))
