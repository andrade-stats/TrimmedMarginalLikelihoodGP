
import sklearn.datasets
import sklearn.preprocessing

import numpy
import sklearn.linear_model
import sklearn.metrics

import simDataGeneration
import generateSyntheticData

import commonSettings

import torch

import random
import os

if not os.path.exists(commonSettings.PREPARED_DATA_FOLDER):
    os.makedirs(commonSettings.PREPARED_DATA_FOLDER)

if not os.path.exists(commonSettings.ALL_RESULTS_FOLDER):
    os.makedirs(commonSettings.ALL_RESULTS_FOLDER)


ALL_NOISE_TYPES = ["noNoise", "focused" , "uniform", "asymmetric"]
# ALL_NOISE_TYPES = ["noNoise"]
# ALL_NOISE_TYPES = ["focused" , "uniform", "asymmetric"]

SPLIT_TRAIN_TEST = True

# datasetName = "spacega" # Election data including spatial coordinates on 3,107 US counties. Used in Pace and Barry (1997), Geographical Analysis, Volume 29, 1997, p. 232-247. Submitted by Kelley Pace (kpace@unix1.sncc.lsu.edu). [3/Nov/99] (548 kbytes)
# datasetName = "dengue_iquitos"
# datasetName = "dengue_san_juan"
# datasetName = "TopGear"
# datasetName = "bodyfat"
# datasetName = "housing"
# datasetName = "cadata"
# datasetName = "syntheticSimpleSin"
datasetName = "Friedman_n100"
# datasetName = "Friedman_n400"
# datasetName = "Friedman_n800"
# datasetName = "syntheticFewDotData"


# specify here the ratio of outliers
#  in UAI paper, we used mostly 0.1 (and for the analysis 0.2, 0.3, and 0.4)
commonSettings.TRUE_OUTLIER_RATIO_FOR_NOISE = 0.10



def splitTrainingAndTest(X, y, testDataRatio):
    
    if testDataRatio > 0.0:
        testDataSize = int(y.shape[0] * testDataRatio)
        
        rndIdOrder = numpy.arange(y.shape[0])
        numpy.random.shuffle(rndIdOrder)
        testDataIds = rndIdOrder[0:testDataSize]
        trainDataIds = rndIdOrder[testDataSize:y.shape[0]] 
    
        X_train = X[trainDataIds, :]
        y_train = y[trainDataIds]
        X_test = X[testDataIds, :]
        y_test = y[testDataIds]
    else:
        X_train = X
        y_train = y
        X_test = numpy.zeros((0,0))
        y_test = numpy.zeros(0)
    
    return X_train, y_train, X_test, y_test



for NOISE_TYPE in ALL_NOISE_TYPES:

    numpy.random.seed(3523421)
    torch.manual_seed(3523421)
    random.seed(3523421)

    # **************************
    if NOISE_TYPE == "noNoise":
        TRUE_OUTLIER_RATIO = 0.0
        noisePostFix = ""
    else:
        TRUE_OUTLIER_RATIO = commonSettings.TRUE_OUTLIER_RATIO_FOR_NOISE
        noisePostFix = "_" + str(int(TRUE_OUTLIER_RATIO * 100))
    # **************************


    print("datasetName = ", datasetName)

    if datasetName.startswith("synthetic") or datasetName.startswith("Friedman"):
        NR_TEST_DATA_SAMPLES = 2000

        if datasetName == "syntheticLinearOrthogonal":
            dataType = "orthogonal"
            n = 400 + NR_TEST_DATA_SAMPLES
            allX, allY, trueBeta, responseStd, correlationMatrix  = simDataGeneration.getSyntheticLinearData(dataType, n, nrRepetitions = commonSettings.NUMBER_OF_FOLDS)
        elif datasetName.startswith("syntheticFewDotData"):
            assert(SPLIT_TRAIN_TEST)
            n  = 5
            allX, allY, trueNoiseVariance  = simDataGeneration.getFewDotData(nrRepetitions = commonSettings.GLOBAL_NUMBER_OF_FOLDS)
        elif datasetName.startswith("Friedman"):
            assert(SPLIT_TRAIN_TEST)
        
            if datasetName == "Friedman_n100":
                n = 100 + NR_TEST_DATA_SAMPLES
            elif datasetName == "Friedman_n400":
                n = 400 + NR_TEST_DATA_SAMPLES
            elif datasetName == "Friedman_n800":
                n = 800 + NR_TEST_DATA_SAMPLES
            else:
                assert(False)
            
            allX, allY, trueNoiseVariance  = simDataGeneration.getFriedmanData(n, nrRepetitions = commonSettings.GLOBAL_NUMBER_OF_FOLDS)
        elif datasetName == "syntheticLinearCorrelated":
            dataType = "correlated"
            n = 400 + NR_TEST_DATA_SAMPLES
            allX, allY, trueBeta, responseStd, correlationMatrix  = simDataGeneration.getSyntheticLinearData(dataType, n, nrRepetitions = commonSettings.NUMBER_OF_FOLDS)
        elif datasetName == "syntheticSimpleSin":
            assert(SPLIT_TRAIN_TEST)
            dataType = "syntheticSimpleSin"
            n = 400 + NR_TEST_DATA_SAMPLES
            NUMBER_OF_VARIABLES = 1
            allX, allY, trueNoiseVariance = simDataGeneration.getSyntheticNonLinearData(dataType, n, NUMBER_OF_VARIABLES, nrRepetitions = commonSettings.GLOBAL_NUMBER_OF_FOLDS)
        elif datasetName == "syntheticLinearCorrelated_large":
            assert(SPLIT_TRAIN_TEST)
            dataType = "correlated"
            n = 4000 + NR_TEST_DATA_SAMPLES
            allX, allY, trueBeta, responseStd, correlationMatrix  = simDataGeneration.getSyntheticLinearData(dataType, n, nrRepetitions = commonSettings.NUMBER_OF_FOLDS)
        elif datasetName == "syntheticSimpleSin_large":
            assert(SPLIT_TRAIN_TEST)
            dataType = "syntheticSimpleSin"
            n = 4000 + NR_TEST_DATA_SAMPLES
            allX, allY = simDataGeneration.getSyntheticNonLinearData(dataType, n, nrRepetitions = commonSettings.NUMBER_OF_FOLDS)
        else:
            assert(False)

    else:
        
        trueNoiseVariance = "unknown"

        if datasetName == "cadata" or datasetName == "bodyfat" or datasetName == "housing" or datasetName == "spacega":
            X_original, y_original = sklearn.datasets.load_svmlight_file("openDatasets/" + datasetName) 
            X_original = X_original.toarray()
        elif datasetName.startswith("dengue"):
            X_original = None
            y_original = None
        else:
            allData_original = numpy.load("openDatasets/" + datasetName + ".npy", allow_pickle = True).item()
            X_original = allData_original["X_original"]
            y_original = allData_original["y_original"]
            

        allX = []
        allY = []
        
        
        
        for foldId in range(commonSettings.GLOBAL_NUMBER_OF_FOLDS):
            allX.append(X_original)
            allY.append(y_original)
                

    if datasetName == "cadata":
        assert(SPLIT_TRAIN_TEST)
        SUB_SAMPLE_SIZE = 800
        n = X_original.shape[0]
        TEST_DATA_RATIO = 1.0 - (SUB_SAMPLE_SIZE / n)
    elif datasetName.startswith("syntheticFewDotData"):
        assert(not SPLIT_TRAIN_TEST)
        TEST_DATA_RATIO = 0.0  
    elif datasetName.startswith("synthetic") or datasetName.startswith("Friedman"):
        TEST_DATA_RATIO = NR_TEST_DATA_SAMPLES / n
    else:
        if SPLIT_TRAIN_TEST:
            TEST_DATA_RATIO = 0.1
        else:
            TEST_DATA_RATIO = 0.0

    assert(TRUE_OUTLIER_RATIO >= 0.0 and TRUE_OUTLIER_RATIO <= 0.5)

    all_X_train = []
    all_y_train = []
    all_X_cleanTest = []
    all_y_cleanTest = []
    all_trueOutlierIndicesZeroOne = []
    all_dataScalerX = []
    all_dataScalerY = []

    for foldId in range(commonSettings.GLOBAL_NUMBER_OF_FOLDS):
        
        print(f"********************** data fold id = {foldId} **********************")

        if datasetName.startswith("dengue"):
            assert(SPLIT_TRAIN_TEST)
            assert(NOISE_TYPE != "noNoise")
            allData_clean = numpy.load(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + "trainTestData" + "_" + "noNoise" + commonSettings.getNoisePostFix("noNoise") + ".npy", allow_pickle = True).item()
            assert(len(allData_clean["all_X_train"]) == 1)
            X_cleanTrain = allData_clean["all_X_train"][0]
            y_cleanTrain = allData_clean["all_y_train"][0]
            X_cleanTest = allData_clean["all_X_cleanTest"][0]
            y_cleanTest = allData_clean["all_y_cleanTest"][0]
                
        else:

            X_original = allX[foldId]
            y_original = allY[foldId]

            X_original = X_original.astype(numpy.float32)
            y_original = y_original.astype(numpy.float32)

            print(f"d = {X_original.shape[1]}, n = {X_original.shape[0]}")
            
            X_cleanTrain, y_cleanTrain, X_cleanTest, y_cleanTest = splitTrainingAndTest(X_original, y_original, TEST_DATA_RATIO)
        

        X_train, y_train, trueOutlierIndicesZeroOne, X_cleanTest, y_cleanTest, dataScalerX, dataScalerY = generateSyntheticData.addNoise_and_scale(X_cleanTrain, y_cleanTrain, X_cleanTest, y_cleanTest, TRUE_OUTLIER_RATIO, NOISE_TYPE)
        
        print("dataset = ", datasetName)
        print("training data size = ", X_train.shape[0])
        print("test data size = ", X_cleanTest.shape[0])
        
        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_cleanTest.append(X_cleanTest)
        all_y_cleanTest.append(y_cleanTest)
        all_trueOutlierIndicesZeroOne.append(trueOutlierIndicesZeroOne)
        all_dataScalerX.append(dataScalerX)
        all_dataScalerY.append(dataScalerY)

    
    allData = {}
    allData["all_X_train"] = all_X_train
    allData["all_y_train"] = all_y_train
    allData["all_trueOutlierIndicesZeroOne"] = all_trueOutlierIndicesZeroOne
    allData["all_dataScalerX"] = all_dataScalerX
    allData["all_dataScalerY"] = all_dataScalerY
    
    if not SPLIT_TRAIN_TEST:
        numpy.save(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + "wholeData" + "_" + NOISE_TYPE + noisePostFix,  allData)
    else:
        allData["all_X_cleanTest"] = all_X_cleanTest
        allData["all_y_cleanTest"] = all_y_cleanTest
        numpy.save(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + "trainTestData" + "_" + NOISE_TYPE + noisePostFix,  allData)

    print("*** successfully saved all data ***")

print("commonSettings.NUMBER_OF_FOLDS = ", commonSettings.GLOBAL_NUMBER_OF_FOLDS)