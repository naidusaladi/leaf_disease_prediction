# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:29:09 2023

@author: DELL

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
import os
from os import listdir
from PIL import Image
from sklearn.preprocessing import label_binarize,  LabelBinarizer
from keras.preprocessing import image
#from keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
model=load_model("C:\\Users\\DELL\\Downloads\\model (1).h5")
# load an image using keras.preprocessing
image_path="C:\\Users\\DELL\\Downloads\\test_dataset\\Tomato___Bacterial_spot\\tomato14.jpg"
img = image.load_img(image_path, target_size=(256, 256))
all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
# convert the image to a numpy array using img_to_array
img_array = img_to_array(img)
ig=[img_array]


x_test = np.array(ig, dtype=np.float16) / 225.0

x_test = x_test.reshape( -1, 256,256,3)
    
print(x_test)
#x_test=convert_image_to_array(image_path)
y_pred = model.predict(x_test)
print(y_pred)
k=np.argmax(y_pred[0])

print(k)

print(all_labels[k])