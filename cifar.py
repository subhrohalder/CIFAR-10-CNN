#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 14:23:07 2020

@author: subhrohalder
"""


#importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from tensorflow import keras




#loading Cifar 10 dataset
cifar10 = keras.datasets.cifar10
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

#label from 
#0. airplane										
#1. automobile										
#2. bird										
#3. cat										
#4. deer										
#5. dog										
#6. frog										
#7. horse										
#8. ship										
#9. truck


#checke the shape of the dataset
X_train.shape
X_test.shape

y_train.shape
y_test.shape


#ploting single image
plt.figure()
plt.imshow(X_train[1])
print(y_train[1])


#plotting the image grid
width_grid = 15
length_grid = 15

fig,axes = plt.subplots(width_grid,length_grid,figsize =(25,25))
axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0,width_grid*length_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')
plt.subplots_adjust(hspace=1)

#converting to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#as we have 10 different classes
no_of_labels = 10

#converting output labels to binary #y_train
y_train = keras.utils.to_categorical(y_train,no_of_labels)
y_train

#converting output labels to binary #y_test
y_test = keras.utils.to_categorical(y_test,no_of_labels)
y_test

#converting the values between 0 to 1
X_train = X_train/255.0
X_test = X_test/255.0

#importing libraries for CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

#model building
cnn_model = Sequential()

cnn_model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu',input_shape = (32,32,3)))
cnn_model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.3))



cnn_model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu'))
cnn_model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.2))

#flattening
cnn_model.add(Flatten())

#adding dense layer
cnn_model.add(Dense(units = 512,activation ='relu'))
cnn_model.add(Dense(units = 512,activation ='relu'))

#as it is classification using activation function as softmax
cnn_model.add(Dense(units = 10,activation ='softmax'))



cnn_model.compile(loss = 'categorical_crossentropy',optimizer = keras.optimizers.RMSprop(lr = 0.001),metrics =[ 'accuracy'])

cnn_model.summary()

#model training use more epochs for better accuracy
history = cnn_model.fit(X_train,y_train,batch_size =32,epochs = 2,shuffle = True)

#checking the accuracy
evaluation = cnn_model.evaluate(X_test,y_test)
print('Test Accuracy:',evaluation[1])

#finding out the predicted classes
predicted_labels = cnn_model.predict_classes(X_test)
predicted_labels

#converting binary to integer
y_test = y_test.argmax(1)

#ploting some predicted labels with image
l = 8
w = 8

fig, axes = plt.subplots(l,w,figsize = (12,12))
axes = axes.ravel()

for i in np.arange(0,l*w):
    axes[i].imshow(X_test[i])
    axes[i].set_title(f'predicted = {predicted_labels[i]} \n actual = {y_test[i]}')
    axes[i].axis('off')
    
plt.subplots_adjust(wspace = 1,hspace=1)


#finding out confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test,predicted_labels)

#ploting confusion matrix
plt.figure(figsize = (10,10))
sns.heatmap(cm,annot = True)

#saving the model
cnn_model.save('cifar10_CNN_model.h5')









































