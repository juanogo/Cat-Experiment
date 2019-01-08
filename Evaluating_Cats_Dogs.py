#!/usr/bin/env python
# coding: utf-8
"""
Created on Sept 2015

@author: Juan Gomez, PhD
"""

# In[93]:


#from keras.models import model_from_json
from tensorflow import keras
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# In[10]:


json_file = open('D:/all/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)

loaded_model.load_weights("D:/all/model.h5")
print("Loaded model from disk")


# In[11]:


loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[54]:


img = cv2.imread("D:/all/train/cat.1.jpg")
img = cv2.resize(img, (50,50))
print(img.shape)
img = img.reshape(1, 50, 50, 3)


# In[50]:


print(loaded_model.predict(img))


# In[141]:


def try_network(dir):
    counter=1
    num_rows = 5
    num_cols = 5
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*num_cols, 2*num_rows))
    for i in os.listdir(dir):
        if np.random.random_sample()>0.01:
            path = os.path.join(dir,i)
            img = cv2.imread(path)
            plt.subplot(num_rows,num_cols, counter)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img, cmap=plt.cm.binary)
            img = cv2.resize(img, (50,50))
            img = img.reshape(1, 50, 50, 3)
            if loaded_model.predict(img) == 0:
                label = 'C A T'
            else:
                label = 'D O G'  
            plt.xlabel(label)
            counter+=1
            if counter>num_images:
                break

        


# In[146]:


try_network('D:/all/examples')


# In[91]:




