# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 00:10:42 2018

@author: Silent Hacker
"""
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding,Bidirectional, Dropout
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np


num_classes = 2; #0,1,2,3(total of 4)
maxlen = 250

x_train = np.load('F:/Programming/Python programming/Prototype1/Dense cell data/x_train.npy')
X_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x = np.array(X_train,dtype=np.float32)
print(x.shape)
#y = np.array([1,1,0,0], dtype=np.int32)

#convert y2 to dummy variables
#y2 = np.zeros((y.shape[0], num_classes),dtype=np.float32)
#y2[np.arange(y.shape[0]),y]=1.0
y2 = np.load('F:/Programming/Python programming/Prototype1/Dense cell data/y_train.npy')
print(y2)
print(y2.shape)
#Nural network model in keras
#model = Sequential()
#model.add(LSTM(128,activation='tanh', recurrent_activation='hard_sigmoid', dropout=0.2, recurrent_dropout=0.2, input_shape=(maxlen,1)))
#model.add(Dense(2,activation='sigmoid'))

model = Sequential()
model.add(Dense(32, input_dim=maxlen,activation='tanh'))
model.add(Dense(64, activation='softmax'))
model.add(Dense(2, activation='sigmoid'))
#try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Training...')
model.fit(x,y2,batch_size=500 ,epochs=2000,verbose=1)
scores = model.evaluate(x, y2)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
pred = model.predict(x)
predict_classes=np.argmax(pred,axis=1)
#print("predict output: ",pred)
print("Predicted classes: {} ",predict_classes)
print("Expected classes: {} ",predict_classes)

#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")

#testing our model
def runit(model, inp):
    inp = np.array(inp,dtype=np.float32)
    pred = model.predict(inp)
    return np.argmax(pred[0])

#val = runit(model,[X_train[1]])
#print("predicted value: ",val)
#if val == 1:
#    print("is expert developer!!!")
#else:
#    print("is novice developer!!!")
    

#######################################################################################################
#x = [ 
#     [[0],[1],[8],[4],[3],[8],[9],[2],[5],[7]],
#     [[8],[9],[0],[2],[3],[8],[7],[7],[5],[6]],
#     [[9],[7],[6],[5],[7],[4],[2],[1],[0],[1]],
#     [[5],[5],[2],[8],[6],[5],[5],[2],[8],[8]],
#     [[3],[6],[1],[0],[6],[0],[8],[6],[8],[6]],
#     [[2],[7],[4],[6],[7],[5],[8],[9],[7],[8]]
#     ]
#x = [ 
#     [0,1,8,4,3,8,9,2,5,7],
#     [8,9,0,2,3,8,7,7,5,6],
#     [5,6,7,4,3,2,3,7,8,6],
#     [1,1,2,4,5,6,7,8,7,8],
#     [5,4,3,5,6,7,8,9,7,1],
#     [0,6,5,4,0,5,4,0,5,4]
#     ]
#x=[
#   [[0,5,4],[8,5,6],[6,5,4],[4,7,3]],
#   [[7,4,2],[5,5,3],[7,6,4],[8,9,0]],
#   [[6,7,0],[0,8,9],[8,1,0],[0,4,6]],
#   [[1,1,0],[4,9,7],[5,6,8],[6,5,3]]
#   ]
#x_train=[
#   [18, 169, 1, 156, 1, 157, 1, 158, 1, 159, 1, 160, 1, 69, 1, 161, 1, 70, 2, 14, 129, 22, 123, 5, 12, 2, 11, 215, 173, 50, 189, 5, 87, 207, 4, 119, 116, 21, 210, 4, 124, 132, 195, 4, 114, 174, 87, 186, 4, 135, 19, 104, 4, 31, 198, 9, 234, 194, 242, 4, 145, 9, 95, 235, 19, 77, 4, 111, 9, 92, 236, 19, 76, 4, 31, 197, 177, 219, 9, 92, 232, 83, 50, 96, 4, 93, 127, 125, 133, 171, 4, 134, 117, 96, 118, 9, 3, 206, 238, 193, 188, 3, 56, 239, 3, 170, 153, 56, 95, 4, 93, 131, 104, 243, 209, 77, 76, 172, 196, 208, 6, 6],
#   [37, 0, 48, 113, 40, 128, 152, 0, 0, 52, 8, 3, 44, 7, 61, 39, 33, 32, 0, 108, 16, 94, 105, 103, 82, 80, 10, 75, 106, 3, 13, 0, 64, 16, 97, 65, 78, 98, 3, 7, 72, 0, 0, 85, 0, 0, 60, 100, 74, 71, 90, 17, 66, 29, 10, 107, 101, 0, 15, 8, 3, 7, 88, 15, 25, 67, 30, 49, 45, 0, 63, 62, 58, 46, 57, 43, 51, 79, 81, 17, 86, 0, 59, 3, 7, 23, 3, 102, 89, 84, 99, 68, 0, 91, 8, 3, 13, 34, 18, 165, 1, 70, 1, 162, 1, 164, 1, 166, 1, 167, 2, 14, 122, 22, 144, 5, 12, 2, 73, 214, 199, 5, 38, 29, 221, 176, 187, 25, 192, 154, 163, 73, 190, 4, 143, 120, 140, 201, 35, 42, 137, 41, 136, 115, 202, 36, 109, 53, 200, 35, 42, 139, 41, 138, 36, 109, 53, 27, 237, 6, 12, 26, 11, 213, 224, 5, 228, 38, 23, 233, 231, 10, 223, 230, 227, 225, 6, 26, 11, 226, 240, 5, 241, 6, 6],
#   [37, 0, 48, 40, 147, 126, 0, 0, 52, 8, 3, 44, 7, 61, 39, 33, 32, 0, 108, 16, 94, 105, 103, 82, 80, 10, 75, 106, 3, 13, 0, 64, 16, 97, 65, 78, 98, 3, 7, 72, 0, 0, 85, 0, 0, 60, 100, 74, 71, 90, 17, 66, 29, 10, 107, 101, 0, 15, 8, 3, 7, 88, 15, 25, 67, 30, 49, 45, 0, 63, 62, 58, 46, 57, 43, 51, 79, 81, 17, 86, 0, 59, 3, 7, 23, 3, 102, 89, 84, 99, 68, 0, 91, 8, 3, 13, 34, 18, 218, 1, 216, 1, 217, 2, 14, 130, 5, 121, 4, 110, 203, 4, 141, 2, 55, 179, 5, 27, 211, 6, 2, 24, 55, 5, 2, 20, 183, 2, 20, 184, 6, 2, 24, 20, 5, 2, 54, 182, 6, 2, 24, 54, 5, 2, 21, 185, 6, 6], 
#   [18, 178, 1, 191, 1, 222, 1, 69, 1, 168, 2, 14, 148, 22, 142, 5, 2, 28, 83, 21, 146, 4, 112, 220, 28, 47, 175, 2, 11, 212, 5, 229, 149, 4, 181, 6, 2, 28, 47, 180, 5, 27, 150, 6, 12, 26, 11, 155, 204, 5, 205, 151, 6, 6]
#  ]