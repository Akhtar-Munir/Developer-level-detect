# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:15:56 2018

@author: Silent Hacker
"""
import numpy as np
import json
import tensorflow as tf
from keras.preprocessing import sequence

X_train = np.load("F:/Programming/Python programming/data sets/x_train.npy")
f = open("F:/Programming/Python programming/data sets/imdb_word_index.json",'r')
word_index = json.load(f)
f.close()
word_dict = {idx: word for word, idx in word_index.items()}
sample = []


#print("before padding: ",X_train)
#X_train = sequence.pad_sequences(X_train, maxlen=250)
#print(len(X_train), 'train sequences')
#print("x data: ",X_train)

for i in range(len(X_train)):
    for idx in X_train[i]:
        sample.append(word_dict[idx])
        ' '.join(sample)

#        print("id: ",X_train[i])
#        print("\nwords: ",sample)
        save_to_file = open("F:/Programming/Python programming/data sets/all_pos_views.txt",'a')
        save_to_file.write(str(X_train[i])+"\n"+str(sample))
        save_to_file.close
        print("done saving idx to words!!")
    sample=[]
