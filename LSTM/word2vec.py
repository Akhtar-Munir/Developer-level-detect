from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import pyprind
import pandas as pd

import numpy as np
import tensorflow as tf
import re
#import matplotlib.pyplot as plt
import json
Py3 = sys.version_info[0] == 3
basepath = 'F:/Programming/Python programming/data sets/practice/'
pbar = pyprind.ProgBar(32)
df = pd.DataFrame()
def _read_words(basepath):
    txt = []
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basepath, s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 
                          'r', encoding='iso-8859-1') as infile:
                    txt = txt+infile.read().replace(",", " ").split()
                    pbar.update()
    return txt

def _dic_to_vector(basepath,word_to_id):
    x_train = []
    x_test = []
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(basepath, s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 
                          'r', encoding='iso-8859-1') as infile:
                    vd = (infile.read().replace(",", " ").split())
                    if(s=='test'):
                        x_test.append(_file_to_word_ids(vd,word_to_id))
                    else:
                        x_train.append(_file_to_word_ids(vd,word_to_id))
    return x_train,x_test

def _build_vocab(data):
#  print ("data in buld_vocab = ",data)

  counter = collections.Counter(data)
  #print("counter in buld_vocab = ",counter)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  #print("\n count_pairs in buld_vocab = ",count_pairs)

  words, _ = list(zip(*count_pairs))
#  print("\nword in buld_vocab = ",words)
  word_to_id = dict(zip(words, range(len(words))))
  return word_to_id


def _file_to_word_ids(data, word_to_id):
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  data=[]
  data = _read_words(data_path)
#  print("data: ",data)
  word_to_id = _build_vocab(data)

#  f = open("F:/Programming/Python programming/data sets/DLD/word_to_id_dict.json",'r')
#  word_to_id = json.load(f)
#  print(len(word_to_id))
#  f.close()
  x_train,x_test = _dic_to_vector(data_path,word_to_id)
  vocabulary = len(word_to_id)
  return x_train,x_test,vocabulary,word_to_id
#--------------------------------------------------------------------------------------------------#
def save_one_hot(vector,dist_path):
    d = vector
    el = np.ones((1,int(len(d)/2)))
    nvl = np.zeros((1,int(len(d)/2)))
    indices = np.append(el,nvl)
    depth = 2
    print("depth: ",depth)
    one_hot1 = tf.one_hot(indices, depth,
           on_value=1.0, off_value=0.0,
           axis=1)
    
    with tf.Session() as sess:
        sess.run(one_hot1)
        data = one_hot1.eval(session = sess)
        
    np.save(dist_path,data)
    print("done writing Ont_hot")
    return indices
#----------------------------------------------------------------------------------------------#    
def save_vector(vector,dist_path):
    np.save(dist_path,vector)
    print("done writing vector file!!!!")
def save_dict(word_to_id,dist_path):
    jsn = json.dumps(word_to_id)
    f = open(dist_path,'w')
    f.write(jsn)
    f.close()
    print("Done writing dict.json!!")
    
def removeComments(string):
    string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurance streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile("//.*?\n" ) ,"" ,string) # remove all occurance singleline comments (//COMMENT\n ) from string
    return string

def toBinary(vector):
    binary = []
    binary_list = []
    for i in range(len(vector)):
        if(vector[i]==0):
            binary_list.append(list([0]))
        else:
            while(vector[i]):
                if(vector[i] % 2 == 0):
                    binary.append(0)
                else:
                    binary.append(1)
                vector[i] = vector[i]//2
            binary.reverse()
            binary_list.append(list(binary))
            binary.clear()  
        return binary_list
    

x_train,x_test, vocabulary, dictionary = ptb_raw_data(basepath)
#print("dictionary: ",dictionary)
#print("\nx_train : ",x_train)
#print("\n\nx_test : ",x_test)
print("vocabulary: ",vocabulary)

#saving dictionary
save_dict(dictionary,"E:/word_to_index.json")
#saving vector
save_vector(x_train,'E:/x_train.npy')
save_vector(x_test,'E:/x_test.npy')
#saving vector to one_hot
np.save("E:/indices_y_train.npy",save_one_hot(x_train,"E:/y_train.npy"))
np.save("E:/indices_y_test.npy",save_one_hot(x_test,"E:/y_test.npy"))



