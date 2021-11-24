#!/usr/bin/python3

# main.py
# ML Apnoe and desaturation detection
#
# Author:   Ing. Martin Barton
# Contact:  ma.barton@seznam.cz
# Date:     2021

from CNN_TK import *
import sys, getopt

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

# Arguments and terminal lunch handler
def args(argv):

    if sys.argv[1] == '-h':
        print("\nTo run detection insert nupmy array file path as argument (e.g.):")
        print('python3 main.py array.npy\n')
        sys.exit()
    else:
        main(sys.argv[1])

if __name__ == "__main__":
   args(sys.argv[1])

