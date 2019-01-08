#!/usr/bin/env python
# coding: utf-8
"""
Created on Sept 2015

@author: Juan Gomez, PhD
"""

# In[2]:


import os  
from tqdm import tqdm
import cv2


# In[140]:


TRAIN_DIR = 'D:/all/train'  
TEST_DIR = 'D:/all/test'
IMG_SIZE = 50


# In[157]:


def create_train_test_dir():
    counter=1
    for img in os.listdir(TRAIN_DIR):
    	word_label = img.split('.')[-3]
    	if word_label == 'cat': lable="cats"
    	elif word_label == 'dog': lable="dogs"
    	path = os.path.join(TRAIN_DIR,img)
    	img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    	img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    	cv2.imwrite("D:/all/train/training_data/"+lable+"/"+str(counter)+".jpg",img)
    	#print("image saved"+str(counter))
    	counter += 1


# In[161]:


def process_val_data():
    counter = 1
    for img in os.listdir(TEST_DIR):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        cv2.imwrite("D:/all/test/val_data/"+str(counter)+'.jpg',img)
        counter += 1
        #print(counter)
 


# In[162]:


process_val_data()


# In[ ]:


create_train_test_dir()

