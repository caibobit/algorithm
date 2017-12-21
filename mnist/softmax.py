#!/usr/bin/python
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data
#import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples // batch_size

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
x = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init=tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver()

# Train
with tf.Session() as sess:
  sess.run(init)
  
  for epoch in range(11):
    for batch in range(n_batch):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
  
    acc =sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
    print("Iter " + str(epoch) + "Text accuracy : " + str(acc))
  saver.save(sess,'/root/tensorflow/save/model.ckpt')
    
