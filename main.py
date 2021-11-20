# main.py
# ML Apnoe and desaturation detection
#
# Author:   Ing. Martin Barton
# Contact:  ma.barton@seznam.cz
# Date:     2021

from CNN_TK import *

# main
# Function that starts detection
# Preprocess consist of > downsampling, filtration, segmentation
#
# INPUTS:
# 	        file_path   = File path fo input data (2D numpy array with flow and SpO2 signal)
#
# OUTPUTS:
#    	    Prints out number of detected Apnoe and Desaturation segments

def main(file_path):
    # Apnea and desaturation decetcion with 2DCNN

    print("Apnoe detection start")
    AP_total = detection(data_array=file_path, event_name="flow", 
                         model_h5="model_apnoe_best_final.h5", fsamp=250, threshold=0.5)
    
    print("SpO2 detection start")
    SP_total = detection(data_array=file_path, event_name="spo2", 
                         model_h5="model_spo2_best_final.h5", fsamp=250, threshold=0.5)


    print("\nFinal results:")
    print("Apnoe (C+O):\t" + str(np.sum(AP_total)))
    print("Desaturations:\t" + str(np.sum(SP_total)))

main("data_short.npy")

