#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 18:12:10 2018

@author: sahil
"""
#Libraries
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#Train
data = pd.read_csv('train.csv')
data = data.iloc[:, 1:2]
data = data.values
#pl.plot(data, color='blue')
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
x_train = data[:-1]
y_train = data[1:]
x_train=np.reshape(x_train,(1257,1,1))

model = Sequential()
model.add(LSTM(output_dim=4, activation="sigmoid", input_shape=[None,1]))
model.add(Dense(output_dim=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=32,nb_epoch=200)
#Test
test = pd.read_csv('test.csv')
test = test.iloc[:,1:2]
test.values
x_test = test
x_test = scaler.transform(x_test)
x_test.shape
x_test = np.reshape(x_test, (20, 1, 1))
predict = model.predict(x_test)
predict = scaler.inverse_transform(predict)
pl.plot(test, color='blue', label="Real")
pl.plot(predict, color='red', label="Predicte")
pl.xlabel("Time")
pl.ylabel("Price")
pl.legend()
pl.show()
