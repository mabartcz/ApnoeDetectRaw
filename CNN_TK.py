# CNN_TK.py
# Convolutional  neural network tool kit
#
# Library with support functions for CNN training and testing.
#
# SEE ALSO:
#           CNN_2D.py
#
# Author:   Bc. Martin Barton
# Contact:  ma.barton@seznam.cz
# Date:     2020-18-05

from keras import backend as K
import numpy as np
import sys

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

def TVT_spitl(X, y, subj, chan, seed=False):
    # TVT_split
    # Split imput data into training/validation/testing datasets
    # Description
    #
    # INPUTS:
    # 	        X =     matrix with 2 signals (flow and spo2)
    #           y =     labels in matrix
    #           subj =  information about subjects (generated from Bigdata_munti.py)
    #           chan =  "flow" / "spo2" / "origin"
    #           seed =  If seed == False, than data are randomly selected, with given seed selection is fixed.
    #
    # OUTPUTS:
    #    	    X_train, X_test, X_valid = signal matrix as imported
    #           y_train, y_test, y_valid = vecrot of lables prior to chan variable
    #
    # BRIEF EXPLANATION: This function will split data to 3 seperate datasets prior to whole subjects. Train-Valid-test
    #                    in percentage 60%/15%/15%. Imput is taken from flow and spo2 but outpus labeling is only from one
    #                    of them based on variable chan.
    #
    # SEE ALSO:
    #           Bigdata_multi.py
    #
    # Author:   Bc. Martin Barton
    # Contact:  ma.barton@seznam.cz
    # Date:     2020-18-05

    # Whole subjects to datasets
    X_train = []
    X_valid = []
    X_test = []
    y_train = []
    y_valid = []
    y_test = []

    train_s = 0
    test_s = 0
    valid_s = 0


    destination = 0
    index_count = 0

    for x in subj:
        if x == 1:
            if seed != False:
                np.random.seed(seed)
            destination = np.random.randint(0,3)

            if destination == 0 or destination == 2:
                if len(y_test) > len(y_valid):
                    destination = 2
                else:
                    destination = 0

            if len(y_test)+len(y_valid) > int((len(y_train)+len(y_test)+len(y_valid))/3):
                destination = 1

            # Watch how many subjects in sets
            if destination == 1:
                train_s += 1
            elif destination == 2:
                valid_s += 1
            elif destination == 0:
                test_s += 1

        if destination == 1:
            X_train.append(X[index_count])
            y_train.append(y[index_count,:,0])
        elif destination == 0:
            X_test.append(X[index_count])
            y_test.append(y[index_count,:,0])
        elif destination == 2:
            X_valid.append(X[index_count])
            y_valid.append(y[index_count,:,0])

        index_count +=1

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_valid = np.array(X_valid)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)

    if chan != "origin":

        if chan == "flow":
            s = 0
        elif chan == "spo2":
            s = 1
        else:
            sys.exit("Wrong channel name")

        # Only s labels are interested
        y_train = y_train[:,s]
        y_test = y_test[:,s]
        y_valid = y_valid[:,s]

    print("Data shapes:")
    print("X_train:\t" + str(X_train.shape))
    print("y_train:\t" + str(y_train.shape))
    print("X_valid:\t" + str(X_valid.shape))
    print("y_valid:\t" + str(y_valid.shape))
    print("X_test: \t" + str(X_test.shape))
    print("y_test: \t" + str(y_test.shape))

    print("Train subjs:\t" + str(train_s))
    print("Valid subjs:\t" + str(valid_s))
    print("Test subjs:\t" + str(test_s))

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def The_answer(real, predict, text, treshold=False):
    # The answer
    # Calculate basic statistics on given labels
    # Description
    #
    # INPUTS:
    # 	        real    = vector with real lables (0/1 int)
    #           predict = vector with predicted labels
    #           text    = simple string whitch will be printed
    #           treshold= threshold for pozitives or negatives, if False, than expected predict to by (0/1)
    #
    # OUTPUTS:
    #    	   sensitivity and specificity
    #
    # BRIEF EXPLANATION:    This function will calculate statistics based on prediction with or without threshold. Rest of
    #                       infomration is printed on screen. It works with 2D or 1D inputs.
    #
    # SEE ALSO:
    #           CNN_2D.py
    #
    # Author:   Bc. Martin Barton
    # Contact:  ma.barton@seznam.cz
    # Date:     2020-18-05


    try:
        # For 2D
        if np.shape(real)[1] == 2:
            if treshold != False:
                for x in range(np.shape(predict)[0]):
                    if predict[x, 0] < treshold:
                        predict[x, 0] = 0
                    else:
                        predict[x, 0] = 1

                    if predict[x, 1] < treshold:
                        predict[x, 1] = 0
                    else:
                        predict[x, 1] = 1

            TP = 0
            TN = 0
            FP = 0
            FN = 0

            for c in range(len(real)):
                if real[c,0] == 1 and predict[c,0] == 1:
                    TP += 1
                elif real[c,0] == 0 and predict[c,0] == 0:
                    TN += 1
                elif real[c,0] == 0 and predict[c,0] == 1:
                    FP += 1
                elif real[c,0] == 1 and predict[c,0] == 0:
                    FN += 1

            for c in range(len(real)):
                if real[c,1] == 1 and predict[c,1] == 1:
                    TP += 1
                elif real[c,1] == 0 and predict[c,1] == 0:
                    TN += 1
                elif real[c,1] == 0 and predict[c,1] == 1:
                    FP += 1
                elif real[c,1] == 1 and predict[c,1] == 0:
                    FN += 1

            print("\n" + str(text))
            print("TP: " + str(TP))
            print("TN: " + str(TN))
            print("FP: " + str(FP))
            print("FN: " + str(FN))
            print("Accuracy:\t" + str(round(((TP + TN) / (TP + TN + FP + FN + K.epsilon())) * 100, 2)) + " %")
            print("F1-score:\t" + str(round(((2 * TP) / (2 * TP + FN + FP + K.epsilon())) * 100, 2)) + " %")
            print("Sensitivity:\t" + str(round(((TP) / (TP + FN + K.epsilon())) * 100, 2)) + " %")
            print("Specificity:\t" + str(round(((TN) / (TN + FP + K.epsilon())) * 100, 2)) + " %")

            return (round(((TP) / (TP + FN + K.epsilon())) * 100, 2)), (round(((TN) / (TN + FP + K.epsilon())) * 100, 2))
    except:
            # For 1D
            if treshold != False:
                for x in range(np.shape(predict)[0]):
                    if predict[x, 0] < treshold:
                        predict[x, 0] = 0
                    else:
                        predict[x, 0] = 1

            TP = 0
            TN = 0
            FP = 0
            FN = 0
            P = 0
            for c in range(len(real)):
                if real[c] == 1 and predict[c] == 1:
                    TP += 1
                elif real[c] == 0 and predict[c] == 0:
                    TN += 1
                elif real[c] == 0 and predict[c] == 1:
                    FP += 1
                elif real[c] == 1 and predict[c] == 0:
                    FN += 1

            print("\n" + str(text))
            print("TP: " + str(TP))
            print("TN: " + str(TN))
            print("FP: " + str(FP))
            print("FN: " + str(FN))
            print("Accuracy:\t" + str(round(((TP + TN) / (TP + TN + FP + FN + K.epsilon())) * 100, 2)) + " %")
            print("F1-score:\t" + str(round(((2 * TP) / (2 * TP + FN + FP + K.epsilon())) * 100, 2)) + " %")
            print("Sensitivity:\t" + str(round(((TP) / (TP + FN + K.epsilon())) * 100, 2)) + " %")
            print("Specificity:\t" + str(round(((TN) / (TN + FP + K.epsilon())) * 100, 2)) + " %")

            return (round(((TP) / (TP + FN + K.epsilon())) * 100, 2)), (round(((TN) / (TN + FP + K.epsilon())) * 100, 2))

