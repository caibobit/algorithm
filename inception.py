#!/usr/bin/ptyhon
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:23:54 2017

@author: caibo
"""
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

class NodeLookup(object):
    def __init__(self):
        label_lookup_path ='inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path='inception_model/imagenet_synset_to_human_label_map'
        self.node_lookup=self.load(label_lookup_path,uid_lookup_path)
    
    def load(self,label_lookup_path,uid_lookup_path):
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        #按行读取数据
        for line in proto_as_ascii_lines:
            line=line.strip('\n')
            parsed_items = line.split('\t')
            #分类标签编号
            uid =parsed_items[0]
            #分类名称
            human_string = parsed_items[1]
            #建立两者之间的映射
            uid_to_human[uid] = human_string
        
        #加载1000个分类标签
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid={}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                #获取分类号
                target_class = int(line.split(':')[1])
            if line.startswith('  target_class_string:'):
                #获取字符串
                target_class_string = line.split(':')[1]
                #建立映射
                node_id_to_uid[target_class]=target_class_string[1:-2]
        #将两个文件建立联系
        node_id_to_name={}
        for key,value in node_id_to_uid.items():
            name = uid_to_human[value]
            node_id_to_name[key] = name
        return node_id_to_name
    #根据编号返回名称
    def id_to_string(self,node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]
#创建图来存放训练好的模型参数
with tf.gfile.FastGFile('inception_model/classify_image_graph_def.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #遍历目录
    for root,dirs,files in os.walk('images/'):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file),'rb').read()
            #jpeg格式的图片
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
            #结果转为1维度
            predictions = np.squeeze(predictions)
            #打印图片信息
            image_path = os.path.join(root,file)
            print (image_path)
            #显示图片
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis("off")
            plt.show()
            #排序
            top_k = predictions.argsort()[-5:][::-1]
            node_lookup = NodeLookup()
            for node_id in top_k:
                human_string =node_lookup[node_id]
                #置信度
                score = predictions[node_id]
                print ('%s (score = %.5f)' % (human_string, score))
