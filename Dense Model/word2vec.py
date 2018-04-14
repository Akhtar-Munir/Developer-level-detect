from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import json

Py3 = sys.version_info[0] == 3

files = ["E:/w2v/test_code1.txt"]
#data=[]
def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace(",", " ").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


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


def _file_to_word_ids(fpath, word_to_id):
  data = _read_words(fpath)
  return [[word_to_id[word]] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  train_path = data_path
  w2v = []
  data=[]
  for f in data_path:
    #data.append(_read_words(f))
    data = data+_read_words(f)
#  word_to_id = _build_vocab(data)
  f = open("E:/w2v/word_to_id_dict.json",'r')
  word_to_id = json.load(f)
  for i in train_path:
      w2v.append(_file_to_word_ids(i, word_to_id))
  vocabulary = len(word_to_id)
  return w2v,vocabulary,word_to_id
#--------------------------------------------------------------------------------------------------#
def save_one_hot(vector):
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
        
    np.save("E:/y_train.npy",data)
    print("done writing Ont_hot")
#----------------------------------------------------------------------------------------------#    
def save_vector(vector):
    np.save('E:/test_code1.npy',vector)
    print("done writing vector file!!!!")
def save_dict(word_to_id):
    jsn = json.dumps(word_to_id)
    f = open('E:/word_to_id_dict.json','w')
    f.write(jsn)
    f.close()
    print("Done writing dict.json!!")

'''def toBinary(vector):
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
    return binary_list'''
    

vector_data, vocabulary, dictionary = ptb_raw_data(files)
print("dictionary: ",dictionary)
print("\nwords to vector : ",vector_data)
print("vocabulary: ",vocabulary)
#saving dictionary
#save_dict(dictionary)
#saving vector
save_vector(vector_data)
#saving vector to one_hot
#save_one_hot(vector_data)



