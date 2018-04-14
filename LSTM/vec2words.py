# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:07:50 2018

@author: Silent Hacker
"""
from keras.preprocessing import sequence
import numpy as np
import json

vector_file_path = "E:/x_test.npy"
dict_file_path = "E:/word_to_index.json"

def idx_to_words(vfp,dfp):
    X_vector = np.load(vfp)
#    print("Pad sequences (samples x time)")
#    X_train = sequence.pad_sequences(X_train, maxlen=250)
#    print(X_train)
    f = open(dfp,'r')
    word_index = json.load(f)
    f.close()
    word_dict = {idx: word for word, idx in word_index.items()}
    sample = []
    for idx in X_vector[4]:
        sample.append(word_dict[idx])
    ' '.join(sample)
#    save_to_file = open("E:/rf_1.txt",'w')
#    save_to_file.write(str(sample))
#    save_to_file.close
#    print("done saving idx to words!!")
    return X_vector[4],sample

idx,text = idx_to_words(vector_file_path,dict_file_path)
print("id: ",idx)
print("\nwords: ",text)