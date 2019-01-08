#!/usr/bin/env python
# coding: utf-8
"""
Created on Sept 2015

@author: Juan Gomez, PhD
"""

# In[19]:


import tensorflow as tf
from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense


# In[21]:


# Initialising the CNN
model = keras.models.Sequential()


# In[22]:


# Convolution
model.add(keras.layers.Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))

# Pooling
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(keras.layers.Conv2D(32, (3, 3), activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(keras.layers.Flatten())

# Full connection
model.add(keras.layers.Dense(units = 128, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))


# In[23]:


# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[25]:


train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# In[27]:


training_set = train_datagen.flow_from_directory('D:/all/train/training_data',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[28]:


model.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_steps = 2000)


# In[29]:


model_json = model.to_json()
with open("D:/all/model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("D:/all/model.h5")
print("saved model..! ready to go.")

