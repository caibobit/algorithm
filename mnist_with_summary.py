# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:34:56 2017

@author: caibo
"""

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
from tensorflow.contrib.tensorboard.plugins import projector
#import input_data
import tensorflow as tf

max_step=1001
image_num=3000
DIR = '/root/tensorflow/'
mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)
#载入图片
embedding=tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False,name='embedding')


batch_size=100

n_batch=mnist.train.num_examples // batch_size

sess = tf.InteractiveSession()
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)
        
# 输入
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784],name='x-input')
    y = tf.placeholder(tf.float32, [None, 10],name='y-input')
#显示图片
with tf.name_scope('input_reshape'):
    image_shaped_input=tf.reshape(x,[-1,28,28,1])
    tf.summary.image('input',image_shaped_input,10)    
#网络层
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]),name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b=tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction=tf.nn.softmax(wx_plus_b)
# Define loss and optimizer
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

sess.run(tf.global_variables_initializer())

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy',accuracy)

#生成metadata文件
if tf.gfile.Exists(DIR+'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR+'projector/projector/metadata.tsv')
with open(DIR+'projector/projector/metadata.tsv','w') as f:
    labels= sess.run(tf.argmax(mnist.test.labels[:],1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

#合并参数
merged=tf.summary.merge_all()

projector_writer =tf.summary.FileWriter(DIR+'projector/projector',sess.graph)
saver=tf.train.Saver()
config=projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR+'projector/projector/metadata.tsv'
embed.sprite.image_path = DIR+'projector/data/'
embed.sprite.single_image_dim.extend([28,28])
projector.visualize_embeddings(projector_writer,config) 


# Train
for i in range(max_step):
    batch_xs,batch_ys=mnist.train.next_batch(batch_size)
    run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary,_=sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata,'step%03d' % i)
    projector_writer.add_summary(summary,i)
    
    if i % 100 ==0:
        acc =sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Lter"+str(i)+', Testing Accuracy :'+ str(acc))

saver.save(sess,DIR+'projector/projector/a_model.ckpt',global_step=max_step)
projector_writer.close()
sess.close()
"""  
with tf.Session as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('/tensorflow/mnist_logs/',sess.graph)  
    for epoch in range(10):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            summary,_=sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
        writer.add_summary(summary,epoch)
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})   
        print("Lter"+str(epoch)+', Testing Accuracy :'+ str(acc))
"""        
