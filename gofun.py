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
#
#t = textgenrnn()
#t.train_from_file("test.txt", num_epochs = 5, save_epochs=1, gen_epochs = 0)
#textgen.generate(1, temperature = 1.0)
#t=textgenrnn('textgenrnn_weights.hdf5')
#t = textgenrnn('itc-titles-1%-epochs500.hdf5')
#t = textgenrnn('itc-titles-epoch100.hdf5')
#t = textgenrnn('itc-titles-epochs100.hdf5')
#t = textgenrnn('120epochs5%itc.hdf5')
t = textgenrnn('itc-titles-1%-epochs1000.hdf5')
#t.train_from_file("test.txt", new_model=True, num_epochs = 1, gen_epochs = 0)
#t.train_from_file("test.txt", num_epochs = 1000, save_epochs = 100, gen_epochs = 0, batch_size = 128)
#t.train_from_file("test.txt", new_model =True, num_epochs = 1, gen_epochs = 0, train_size=0.8, dropout=0.2, set_validation=False)
t.generate(5, temperature = 0.8)
#t.generate_samples(prefix="Tesla", max_gen_length=120)
#t = textgenrnn()
#t.reset()
#t.save('itc-titles-1%-epochs1000.hdf5')
#texts = ['Never gonna give you up, never gonna let you down',
#        'Never gonna run around and desert you',
#        'Never gonna make you cry, never gonna say goodbye',
#        'Never gonna tell a lie and hurt you']
#
#
#t.train_on_texts(texts, num_epochs=2,  gen_epochs=2)
#generated_texts = textgen.generate(n=5, prefix="Trump", temperature=0.2, return_as_list=True)
#print(generated_texts)
#textgen.generate_to_file('textgenrnn_texts.txt', n=5)
#textgen.generate(5, prefix="Apple")
#textgen = textgenrnn('../weights/hacker_news.hdf5')
#textgen.generate_samples(temperatures=[0.2, 0.5, 1.0, 1.2, 1.5])
#textgen.reset()
#textgen.train_from_largetext_file(fulltext_path, new_model=True, num_epochs=1,
#                                  word_level=True,
#                                  max_length=10,
#                                  max_gen_length=50,
#                                  max_words=5000)