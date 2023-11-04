import numpy as np
import matplotlib.pyplot as plt
import os
import random
import itertools
import cv2
import requests

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from keras.utils import to_categorical
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from flask import Flask,request,jsonify,render_template

#Function to perform white balancing on images
def white_balance(channel, perc = 0.05):
    mi, ma = (np.percentile(channel, perc), np.percentile(channel,100.0-perc))
    channel = np.uint8(np.clip((channel-mi)*255.0/(ma-mi), 0, 255))
    return channel

#Function to perform CLAHE on images
def clahe():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    return clahe

#Function to enhance images using CLAHE and white balancing.
def image_enhancer(image_arrays):
    enhanced_images = []
    
    for image in image_arrays:
        
        #White Balance
        image_WB  = np.dstack([white_balance(channel, 0.05) for channel in cv2.split(image)] )
        gray_image = cv2.cvtColor(image_WB, cv2.COLOR_RGB2GRAY)

        #CLAHE
        clahe_function = clahe()
        image_clahe = clahe_function.apply(gray_image)
        image = cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2RGB)
        
        enhanced_images.append(image)
    
    return enhanced_images

#Function to normalize dataset
def normalizer(image_arrays):
    # Create an empty list to store normalized arrays
    norm_image_arrays = []
    
    # Iterate over all the image arrays and normalize them before storing them into our predefined list
    for image_array in image_arrays:
        norm_image_array = image_array / 255.0
        norm_image_arrays.append(norm_image_array)
    
    return norm_image_arrays

def upload(directory,size=(224,224)):
    image_array=[]
    image = cv2.imread(directory)
    image = cv2.resize(image, size)
    image_array.append(image)    
    image_enhanced=image_enhancer(image_array)
    image_normalized=normalizer(image_enhanced)
    image_np=np.array(image_normalized)
    class_names = {0: "Normal", 1 : "Viral Pneumonia", 2: "COVID-19"}
    predictions = model.predict(image_np)
    predicted_labels = np.argmax(predictions, axis=1)
    output=(class_names[predicted_labels[0]])
    return output



app=Flask(__name__)
model =load_model("model_tuned.h5")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():    
    return render_template('about.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/output')
def output():
    return render_template('output.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    file = request.files['image']
    if file:
       # Define the directory where you want to save the uploaded audio files
       upload_folder = 'uploads'

       # Ensure the 'uploads' directory exists
       os.makedirs(upload_folder, exist_ok=True)

       # Save the uploaded audio file to the specified directory
       file.save(os.path.join(upload_folder, file.filename))
       temp_path=os.path.join(upload_folder,file.filename)
       output=upload(temp_path)
       
       return render_template('output.html', text="{}".format(output))
       

if __name__ == '__main__':
    app.run(debug=True)