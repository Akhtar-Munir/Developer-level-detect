# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 12:04:01 2018

@author: Silent Hacker
"""
import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
from keras.preprocessing import sequence

#loading model
def load_model():
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #load woeights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return loaded_model
#input code for testing
x_train = np.load('F:/Programming/Python programming/Prototype1/LSTM cell data/x_train.npy')
X_train = sequence.pad_sequences(x_train, maxlen=250)
print(len(X_train), 'train sequences')
#print("x data: ",X_train)

#testing and prediction model
def runit(model, inp):
    inp = np.array(inp,dtype=np.float32)
    pred = model.predict(inp)
    return np.argmax(pred[0])

#load_java_file('E:/x_train.npy')
val = runit(load_model(),[X_train[4]])
print("predicted value: ",val)
if val == 1:
    print("is expert developer!!!")
else:
    print("is novice developer!!!")