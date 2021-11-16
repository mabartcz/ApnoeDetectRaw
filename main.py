# AD_lunch.py
#
# Author:   Ing. Martin Barton
# Contact:  ma.barton@seznam.cz
# Date:     2021

import os
from shutil import copy
from dfileTK import *
from CNN_TK import *
import numpy as np
import traceback
from keras.models import load_model
from keras_sequential_ascii import keras2ascii

def detection(data_array, event_name, model_h5, fsamp, threshold=0.5):
    # Provede detekci flow/spo2 (jen jednoho) ale z obou kanalu
    # event_name = "flow" nebo "spo2"
    # model_h5 = model natrenovane NN ve formate h5
    # threshold = prah pro klasifikator

    if event_name != "flow" and event_name != "spo2":
        print("Wrong event name !")
        sys.exit()

    # Preprocesing dat flow
    data_seg_f = preprocess_flow_slide(data_array, fsamp)

    # Preprocesing dat spo2
    data_seg_s = preprocess_spo2_slide(data_array, fsamp)

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

    print("::::::::::::::::::::::::::::::::::::::::")
    print(X_data.shape)
    print("::::::::::::::::::::::::::::::::::::::::")

    print("Zacatek detekce apnoe.")
    y_data = model.predict(X_data, 1000, verbose=1)

    # Treshold decide
    for x in range(np.shape(y_data)[0]):
        if y_data[x, 0] < threshold:
            y_data[x, 0] = 0
        else:
            y_data[x, 0] = 1
    
    # if event_name == "flow":
    #     The_answer(data_tags_f[:,0], y_data, "NN vesmir, flow")

    #     # vlozi do puvodnich tagu (s umistenim) jak se predikovalo
    #     data_tags_f[:, 0] = y_data[:, 0]

    #     # Prevedeni dat k ulozeni
    #     tags_f = tag_seg2tags_slide(data_tags_f, "O+", "O-")

    #     # spocita celkove pocty detekovanych udalost
    #     EV_total = int(np.shape(tags_f)[0] / 2)

    # elif event_name == "spo2":
    #     The_answer(data_tags_s[:,0], y_data, "NN vesmir, spo2")

    #     # vlozi do puvodnich tagu (s umistenim) jak se predikovalo
    #     data_tags_s[:, 0] = y_data[:,0]

    #     # Prevedeni dat k ulozeni
    #     tags_s = tag_seg2tags_slide(data_tags_s, "D+", "D-")

    #     # spocita celkove pocty detekovanych udalost
    #     EV_total = int(np.shape(tags_s)[0]/2)

    #return EV_total

def preprocess_flow_slide(data_array, fsamp):

        # Nacte kanal spo2 a vyhodi ho nasegmentovanej
        prekryv = 9                         # kolik sec signalu se bude prekryvat
        seg_len_sec = 10                    # Delka segmentu v sec

        # Nacteni dat
        data = np.load(data_array)
        print(data.shape)
        flow_orig = data[0,:]  # Nacteni jednoho kanalu (flow)

        seg_len = fsamp * seg_len_sec   # Delka segmentu ve vzorkach
        bucket_size = seg_len           # Velikost segmentu
        overlap_count = prekryv * fsamp # Prekryv

        # Podvzorkovani
        flow_down = sig.decimate(flow_orig, 5, zero_phase=True)
        fsamp_new = int(fsamp/5)

        # Prenastaveni hodnot po podvzorkovani
        seg_len = fsamp_new * seg_len_sec    # Delka segmentu ve vzorkach
        bucket_size = seg_len                # Velikost segmentu
        overlap_count = prekryv * fsamp_new  # Prekryv

        # Filtrace
        sos = sig.butter(10, 5 / (fsamp_new / 2), btype='lowpass', output='sos')
        flow = scipy.signal.sosfilt(sos, flow_down)

        # Segmentace sliging window
        slider = Slider(bucket_size, overlap_count)
        slider.fit(flow)
        print("Segmentation")
        data = []

        while True:
            window_data = slider.slide()
            if len(window_data) < bucket_size:
                break
            data.append(window_data)
            if slider.reached_end_of_list(): break

        # Konverze na numpy array
        flow_seg = np.array(data)

        return flow_seg

def preprocess_spo2_slide(data_array, fsamp):
        # Nacte kanal spo2 a vyhodi ho nasegmentovanej
        prekryv = 9                         # kolik sec signalu se bude prekryvat
        seg_len_sec = 10                    # Delka segmentu v sec

        #Nacteni dat
        data = np.load(data_array)
        print(data.shape)
        spo2_orig = data[1,:]  # Nacteni jednoho kanalu (SpO2)

        # Nastaveni
        seg_len = fsamp * seg_len_sec  # Delka segmentu ve vzorkach

        bucket_size = seg_len               # Velikost segmentu
        overlap_count = prekryv * fsamp      # Prekryv

        # Podvzorkovani
        spo2_down = sig.decimate(spo2_orig, 5, zero_phase=True)
        fsamp_new = int(fsamp/5)

        # Prenastaveni hodnot po podvzorkovani
        seg_len = fsamp_new * seg_len_sec  # Delka segmentu ve vzorkach
        bucket_size = seg_len               # Velikost segmentu
        overlap_count = prekryv * fsamp_new      # Prekryv

        # Filtrace
        sos = sig.butter(2, 0.06 / (fsamp_new / 2), btype='lowpass', output='sos')
        spo2_filt = scipy.signal.sosfilt(sos, spo2_down)

        # Nasobeni
        spo2 = np.multiply(np.array(spo2_filt), 10)

        # Segmentace sliging window
        slider = Slider(bucket_size, overlap_count)
        slider.fit(spo2)
        print("Segmentation")
        data = []

        while True:
            window_data = slider.slide()
            if len(window_data) < bucket_size:
                break
            data.append(window_data)
            if slider.reached_end_of_list(): break

        # Konverze na numpy array
        spo2_seg = np.array(data)

        return spo2_seg

def play():
    # Zkouska detekce bez GUI
    print("Loading data")
    file_path = "data_raw.npy"
    print("Data loaded")

    #file_path = "/Users/mabarton/SW/detekce-apnoe/data/pokoj2-psg_2016-03-01-2203.d"
    #file_path = "/Volumes/GLaDOS/_DATA/TESTOVACI_data_PSG/ST_7_psg.d"

    print("Apnoe detection start")
    AP_total = detection(data_array=file_path, event_name="flow", 
                         model_h5="model_apnoe_best_final.h5", fsamp=250, threshold=0.5)
    
    # print("SpO2 detection start")
    # SP_total = detection(data_array=file_path, event_name="spo2", 
    #                      model_h5="model_spo2_best_final.h5", fsamp=500, threshold=0.5)


    print("\nFinal results:")
    print("Apnoe (C+O):\t" + str(AP_total))
    # print("Desaturations:\t" + str(SP_total))

play()



