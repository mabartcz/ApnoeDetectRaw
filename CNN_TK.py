# CNN_TK.py
# Convolutional  neural network tool kit
#
# Library with support functions for Apnoe detection wiht CNN
#
# Author:   Ing. Martin Barton
# Contact:  ma.barton@seznam.cz
# Date:     2021

import sys
import scipy
import numpy as np
import numpy as np
import scipy.fftpack
import scipy.signal as sig

from CNN_TK import *
from keras import backend as K
from window_slider import Slider
from keras.models import load_model
from keras_sequential_ascii import keras2ascii

#-------------------------------------------------------------------------------------------------------------
# Metrics
# One line description
# Description
#
# INPUTS:
# 	        y_true = vector with ground true of classification
#           y_pred = vector with predicted values
#
# OUTPUTS:
#    	    Calculated statistival value
#
# SEE ALSO:
#           Keras documentation
#
# Author:  Past Keras verions
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
#-------------------------------------------------------------------------------------------------------------

# Detection
# Function that handles CNN detection on given data
# Description
#
# INPUTS:
# 	        data_array  = 2D numpy array of data (1. row flow, 2. row SpO2)
#           event_name  = "flow", "spo2" depends on what detection is used
#           model_h5    = ML keras model in h5 file
#           fsamp       = sampling frequenci (only supports 250)
#           treshold    = deteciont desizion treshold
#
# OUTPUTS:
#    	    Returns array with values 0 (event not present in segment), 1 (event is present in segment)
#           To corredponding segments. 
def detection(data_array, event_name, model_h5, fsamp, threshold=0.5):

    if event_name != "flow" and event_name != "spo2":
        print("Wrong event name !")
        sys.exit()

    # Preprocesing flow data
    data_seg_f = preprocess_slide(data_array, fsamp, event_name)

    # Preprocesing SpO2 data
    data_seg_s = preprocess_slide(data_array, fsamp, event_name)

    spo2_delay = 25

    # Shape input data
    G = np.shape(data_seg_f)[0] - spo2_delay
    data_seg_f  = data_seg_f[0:G, :]
    data_seg_s  = data_seg_s[spo2_delay - 1:-1, :]

    # Connect flow and spo2
    X_data = np.stack((data_seg_f, data_seg_s), axis=1)

    # Dependencies for kears model
    dependencies = {'sensitivity': sensitivity, 'specificity': specificity, 'f1_score'   : f1_score}

    # Loading of keras model
    model = load_model(model_h5, custom_objects=dependencies)
    keras2ascii(model)
    print("Nacten Keras model")

    print("Dimension expand")
    X_data = np.expand_dims(X_data, 3)

    print("Zacatek detekce apnoe.")
    y_data = model.predict(X_data, 1000, verbose=1)

    # Treshold decide
    for x in range(np.shape(y_data)[0]):
        if y_data[x, 0] < threshold:
            y_data[x, 0] = 0
        else:
            y_data[x, 0] = 1

    return y_data
#-------------------------------------------------------------------------------------------------------------

# Preprocess_slide
# Function that preprocces data
# Preprocess consist of > downsampling, filtration, segmentation
#
# INPUTS:
# 	        data_array  = 2D numpy array of data (1. row flow, 2. row SpO2)
#           event_name  = "flow", "spo2" depends on what detection is used
#           fsamp       = sampling frequenci (only supports 250)
#
# OUTPUTS:
#    	    Returns array with preproccesed, segmented data
def preprocess_slide(data_array, fsamp, event_name):

    overlap = 9                         # How much sec of signal will overlap
    seg_len_sec = 10                    # Segment length in seconds

    # Data loading
    data = np.load(data_array)
    print(data.shape)

    if event_name == "flow":
        data_orig = data[0,:]           # One channel loadnig (flow)
    elif event_name == "spo2":
        data_orig = data[1,:]           # One channel loadnig (SpO2)

    seg_len = fsamp * seg_len_sec       # Segment length in samples
    bucket_size = seg_len               # Size of segment
    overlap_count = overlap * fsamp     # Overlap

    # Downsampling
    data_down = sig.decimate(data_orig, 5, zero_phase=True)
    fsamp_new = int(fsamp/5)

    # New settings after downsamplig
    seg_len = fsamp_new * seg_len_sec    # Segment length in samples
    bucket_size = seg_len                # Size of segment
    overlap_count = overlap * fsamp_new  # Overlap

    if event_name == "flow":
        # Filtration
        sos = sig.butter(10, 5 / (fsamp_new / 2), btype='lowpass', output='sos')
        data_filtred = scipy.signal.sosfilt(sos, data_down)
    
    elif event_name == "spo2":
        # Filtration
        sos = sig.butter(2, 0.06 / (fsamp_new / 2), btype='lowpass', output='sos')
        data_filtred = scipy.signal.sosfilt(sos, data_down)

        # Multiplication
        data_filtred = np.multiply(np.array(data_filtred), 10)

    # Segmentation sliging window
    slider = Slider(bucket_size, overlap_count)
    slider.fit(data_filtred)
    print("Segmentation")
    data = []

    while True:
        window_data = slider.slide()
        if len(window_data) < bucket_size:
            break
        data.append(window_data)
        if slider.reached_end_of_list(): break

    # To numpy array
    data_seg = np.array(data)

    return data_seg
#-------------------------------------------------------------------------------------------------------------
