#!usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:49:08 2017

@author: caibo
"""
import tensorflow as tf
import os
import random
import math
import sys
import Image
import numpy as np

#划分验证集训练集
_NUM_TEST = 500
#random seed
_RANDOM_SEED = 0
#数据集路径
DATASET_DIR = '/root/randomcaptcha/captcha/images/'
#tfrecord存放地址
TFRECORD_DIR = '/root/randomcaptcha/captcha/'

#判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train','test']:
        output_filename =os.path.join(dataset_dir,split_name+'.tfrecords')
        if not tf.gfile.Exists(output_filename):
            return False
    return True
#获取验证码图片
def _get_filenames_and_classes(dataset_dir):
    #图片名称
    photo_filenames = []
    #循环分类的文件夹
    for filename in os.listdir(dataset_dir):
        #获取验证码图片的文件路径
        path = os.path.join(dataset_dir,filename)
        #将图片加入图片列表中
        photo_filenames.append(path)
    #返回结果
    return photo_filenames

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

#图片转换城tfexample函数
def image_to_tfexample(image_data,label0,label1,label2,label3):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': bytes_feature(image_data),
        'label0': int64_feature(label0),
        'label1': int64_feature(label1),
        'label2': int64_feature(label2),
        'label3': int64_feature(label3),
    }))

#数据转换城tfrecorad格式
def _convert_dataset(split_name,filenames,dataset_dir):
    assert split_name in ['train','test']
    
    with tf.Session() as sess:
        #定义tfrecord的路径名字
        output_filename = os.path.join(TFRECORD_DIR,split_name+'.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i ,filename in enumerate(filenames):
                try:
                    sys.stdout.write('\r>> Converting image %d/%d'% (i+1,len(filenames)))
                    sys.stdout.flush()
                    #读取图片
                    image_data = Image.open(filename)
                    #根据模型重新定义结构resize
                    image_data = image_data.resize((224,224))
                    #灰度化图片 
                    image_data = np.array(image_data.convert('L'))
                    #将图片转换成字节bytes
                    image_data = image_data.tobytes()
                    
                    #获取标签
                    labels = filename.split('/')[-1][0:4]
                    nums_labels = []
                    for j in range(4):
                        nums_labels.append(int(labels[j]))
                    
                    #生成tfrecords数据
                    example = image_to_tfexample(image_data,nums_labels[0],nums_labels[1],nums_labels[2],nums_labels[3])
                    tfrecord_writer.write(example.SerializeToString())
                except IOError  as e:
                    print ('could not read:',filenames[1])
                    print ('erroe:' , e)
                    print ('skip it \n')
    sys.stdout.write('\n')
    sys.stdout.flush()

#判断tfrecord文件是否存在
if _dataset_exists(TFRECORD_DIR):
    print ('tfrecord exists')
else:
    #获取图片以及分类
    photo_filenames = _get_filenames_and_classes(DATASET_DIR)
    #随机打乱验证码并划分测试训练集
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_TEST:]
    testing_filenames = photo_filenames[:_NUM_TEST]
    #数据转换
    _convert_dataset('train',training_filenames,DATASET_DIR)
    _convert_dataset('test',testing_filenames,DATASET_DIR)
    print('finish tfrecords!')
