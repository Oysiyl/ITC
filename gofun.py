#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 14:54:12 2018

@author: dmitriy
"""

#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(sess)
#import tensorflow as tf
#with tf.device('/gpu:0'):
#    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#    c = tf.matmul(a, b)
#
#with tf.Session() as sess:
#    print (sess.run(c))
from textgenrnn import textgenrnn

#textgen = textgenrnn()
#t.train_from_file("test.txt", num_epochs=100)
#textgen.generate(1, temperature = 1.0)
t=textgenrnn('textgenrnn_weights.hdf5')
t.generate(5, temperature = 1.0)
#textgen.generate_samples(prefix="low life")
#textgen = textgenrnn()
#textgen.reset()
#texts = ['Never gonna give you up, never gonna let you down',
#        'Never gonna run around and desert you',
#        'Never gonna make you cry, never gonna say goodbye',
#        'Never gonna tell a lie and hurt you']
#
#
#textgen.train_on_texts(texts, num_epochs=2,  gen_epochs=2)