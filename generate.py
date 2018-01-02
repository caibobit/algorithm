#!usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:49:08 2017

@author: caibo
"""
from captcha.image import ImageCaptcha
import numpy as np
import Image
import random
import sys

number = ['0','1','2','3','4','5','6','7','8','9']
#使用alphabet也可以
def random_captcha_text(char_set=number,captcha_size=4):
    #验证码列表
    captcha_text = []
    for i in range(captcha_size):
        #随机选择
        c=random.choice(char_set)
        #加入列表
        captcha_text.append(c)
    return captcha_text
#生成字符对应的验证码
def gen_captchar_text_and_image():
    image = ImageCaptcha()
    #获取随机验证码
    captcha_text = random_captcha_text()
    #验证码转成字符串
    captcha_text = ''.join(captcha_text)
    #生成验证码
    captcha = image.generate(captcha_text)
    #写入文件
    image.write(captcha_text,'captcha/images/'+ captcha_text+'.jpg')
#主函数
num = 1000
if __name__ == '__main__':
    for i in range(num):
        gen_captchar_text_and_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i+1 , num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print ("finished")
