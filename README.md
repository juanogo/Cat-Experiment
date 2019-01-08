# Cat-Experiment
![img_ep_hubel-weisel-toys2](https://user-images.githubusercontent.com/38761819/50806715-cbe2b380-12c5-11e9-8b82-42c338770327.jpg)

For my students in the last course of Machine Learning, I reproduced the results of Nobel Prize winning “Hubel and Wiesel Cat Experiment” 

## The experiment: 

In short, they implanted electrodes in the brain of an anesthetized cat to prove that some neurons in the primary visual cortex process information exclusively form specific patches of the visual field. For instance, one single neuron may be devoted to process information within a single small area at the bottom-left of the visual field and thus, it is only responsive to specific light patterns within such small area. 

A video of the famous experiment that significantly boosted our understanding of visual perception in the 70s, can be watched here: 
https://www.youtube.com/watch?v=RSNofraG8ZE

## Reproduction using Machine Learning:

To reproduce this experiment in silicon, I used for a cat’s brain representation the convolutional neural network model trained on a data set of images to tell cats from dogs (using this image data https://www.kaggle.com/c/dogs-vs-cats). 

Once the model was trained, I used the method described in “Visualizing and Understanding Convolutional Networks” (https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) to find the most responsive neurons in the first layer of the convolutional network (primary cortex) and their associated unique portion in the visual field (an image). Then, I visualized the patterns that when shown up in this portion of the image got to excite the corresponding neuron the most. 

![result](https://user-images.githubusercontent.com/38761819/50806553-3515f700-12c5-11e9-8a17-84278255f52a.png)

## The code is simple:

### DataPreprocessing_Cats_Dogs.py: 
    Preprocesses the images turning them into gray scale as well as re-scaling them down to 50x50 pixels.

### Training_Cats_Dogs.py: 
    Uses a convolutional neural network (with Keras) to tell dogs from cats and save the final model for later usage. 

### Evaluating_Cats_Dogs.py: 
    If the trained model wants to be tried, this file load the saved model above and do the job. I also provided my trained model in case       training a new model wants to be avoided.

### VisualCortexCatExperiment.py: 
    This files finds the most excitable neurons in the first layer, their portion of the visual field and the excitatory patterns. It also     performs a simple visualization of all this. 

PS. Not much comment is given (just enough) into the code so the students can do a better job. 
