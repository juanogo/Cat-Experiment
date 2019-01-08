#!/usr/bin/env python
# coding: utf-8
"""
Created on Sept 2015

@author: Juan Gomez, PhD
"""

# In[4]:


from tensorflow import keras
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# In[5]:


json_file = open('D:/all/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)

loaded_model.load_weights("D:/all/model.h5")
print("Loaded model from disk")


# In[6]:


# Initialising the CNN
model2 = keras.models.Sequential()
# Convolution
model2.add(keras.layers.Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))


# In[7]:


#transfer weoghts of the first layer of the trained model to my one-layer model (biases too)
model2.set_weights(loaded_model.get_weights()[0:1])


# In[8]:


#counts how many times each neuron gets excited with the images in dir
dir='D:/all/test'
maximus=np.zeros((1,48,48,32))
for i in os.listdir(dir):
    path = os.path.join(dir,i)
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (50,50))
        img = img.reshape(1, 50, 50, 3)
        res=model2.predict(img)
        maximus[np.unravel_index(np.argmax(res, axis=None), res.shape)]+=1
    except:
        pass


# In[9]:


#find the 3 (num_neurons) most excited neurons 
excited=list()
patches=list()
count=list()
num_neurons=4
patches  = [list()          for i in range(num_neurons)   ]
for i in range(num_neurons):
    excited.append(np.unravel_index(np.argmax(maximus, axis=None), maximus.shape))
    maximus[np.unravel_index(np.argmax(maximus, axis=None), maximus.shape)]=0
    count.append(0)

#create an image to show the excitatory field (vis_camp) for each of the 3 most excitable neurons
for i in range(num_neurons): 
    vis_camp=np.zeros((50,50))
    vis_camp[excited[i][1]:excited[i][1]+3,excited[i][2]:excited[i][2]+3]=250
    patches[i].append(vis_camp)


# In[10]:


#find the patches with which the most excited neurons get excited
for i in os.listdir(dir):
    path = os.path.join(dir,i)
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (50,50))
        img = img.reshape(1, 50, 50, 3)
        res=model2.predict(img)
        ind=np.unravel_index(np.argmax(res, axis=None), res.shape)
        for x in range(num_neurons):
            if ind==excited[x]and count[x]<8:
                patches[x].append(img[0,ind[1]:ind[1]+3,ind[2]:ind[2]+3,0 ])
                count[x]+=1
    except:
        pass

    


# In[12]:


#it does the plotting thing
plt.figure(figsize=(10, num_neurons))
for i in range(num_neurons):    
 counter=i*10+1
 for x in range(len(patches[i])):
     plt.subplot(num_neurons,10, counter)
     plt.grid(False)
     plt.xticks([])
     plt.yticks([])
     plt.imshow(patches[i][x], cmap=plt.cm.binary)
     counter+=1
     if i==num_neurons-1:
         if counter==i*10+2:
             plt.xlabel('Field')
         elif counter==i*10+6:
             plt.xlabel('...Patterns...')
             


# In[ ]:




