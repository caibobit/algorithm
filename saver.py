#!/usr/bin/python


import tensorflow as tf

log_dir = '/root/tensorflow/train' 

saver = tf.train.import_meta_graph(log_dir+"/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess,log_dir+"/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("w_conv1:0")))
