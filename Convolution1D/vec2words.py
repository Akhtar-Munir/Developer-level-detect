# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:07:50 2018

@author: Silent Hacker
"""
import numpy as np
import json

vector_file_path = "E:/test_code1.npy"
dict_file_path = "E:/w2v/word_to_id_dict.json"

def idx_to_words(vfp,dfp):
    X_train = np.load(vfp)
    print(X_train)
    f = open(dfp,'r')
    word_index = json.load(f)
    f.close()
    word_dict = {idx: word for word, idx in word_index.items()}
    sample = []
    for idx in X_train[0]:
        sample.append(word_dict[idx])
    ' '.join(sample)
#    save_to_file = open("E:/rf_1.txt",'w')
#    save_to_file.write(str(sample))
#    save_to_file.close
#    print("done saving idx to words!!")
    return X_train[0],sample

idx,text = idx_to_words(vector_file_path,dict_file_path)
print("id: ",idx)
print("\nwords: ",text)