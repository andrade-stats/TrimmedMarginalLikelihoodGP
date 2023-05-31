from enum import Enum
import torch
import numpy

EstimationType = Enum("EstimationType", ["FULL_INNER_CV", "LIKELIHOOD_NOISE", "SIMPLE_CV_WITH_FINITE_CORRECTION", "SIMPLE_CV_ONLY", "NO_CORRECTION"])

PREPARED_DATA_FOLDER = "openDatasets_prepared/"
ALL_RESULTS_FOLDER = "all_results/"

ALL_BASELINE_METHODS = ["vanilla","gamma", "student"]

GLOBAL_NUMBER_OF_FOLDS = 10   # 10 should be used for the final experiments

# true ratio of outliers (used for evaluation)
TRUE_OUTLIER_RATIO_FOR_NOISE = None 

# cpu or cuda
DEVICE = None

# data type for tensor 
DATA_TYPE = "float"


def setDevice():

    global DEVICE

    if torch.cuda.is_available():
        DEVICE = "cuda"
        if DATA_TYPE == "double":
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        DEVICE = "cpu"

    return

def assertOnDevice(A):
    if DEVICE == "cuda":
        assert(A.is_cuda)
    elif DEVICE == "cpu":
        assert(A.is_cpu)
    else:
        assert(False)

    return


def setDataType(A):

    if DATA_TYPE == "double":
        A = A.double()
    else:
        assert(DATA_TYPE == "float")
        A = A.float()

    if DEVICE == "cuda":
        return A.cuda()
    else:
        return A.cpu()


def getTorchTensor(A):
    if type(A) is numpy.float32:
        A = torch.tensor(A)
    else:
        A = torch.from_numpy(A)

    A = setDataType(A)
    
    A = A.to(device=DEVICE)
    return A


def getNrFolds(datasetName, noise_type, data_split):

    if datasetName.startswith("dengue"):
        return 1
    elif noise_type == "noNoise" and data_split == "wholeData":
        return 1
    else:
        return GLOBAL_NUMBER_OF_FOLDS


def getNoisePostFix(NOISE_TYPE):
    if NOISE_TYPE == "noNoise":
        noisePostFix = ""
    else:
        noise_percentage = int(TRUE_OUTLIER_RATIO_FOR_NOISE * 100)
        assert(noise_percentage == 10 or noise_percentage == 20 or noise_percentage == 30 or noise_percentage == 40)
        assert(noise_percentage >= 1 and noise_percentage <= 40)
        noisePostFix = "_" + str(noise_percentage)

    return noisePostFix


def getTrueOutlierRatio(NOISE_TYPE):
    if NOISE_TYPE == "noNoise":
        return 0.0
    else:
        return TRUE_OUTLIER_RATIO_FOR_NOISE


def getMethodPostFix(method, OPTIMIZATION_MODE, PRE_SPECIFIED_NU):
    if method.startswith("trimmed_informative_nu"):
        return method + "_" + str(int(PRE_SPECIFIED_NU * 100)) + "nu_" + OPTIMIZATION_MODE
    elif method.startswith("trimmed"):
        return method + "_" + OPTIMIZATION_MODE
    else:
        return method


def getSigmaEstimationTypes():
    return [EstimationType.LIKELIHOOD_NOISE]


def getLabelName(method):
    if method.startswith("trimmed"):
        return r'$\nu$' + "-GP"
    elif method == "student":
        return r'$t$' + "-GP"
    elif method == "gamma":
        return r'$\gamma$' + "-GP"
    elif method == "vanilla":
        return "GP"
    else:
        assert(False)

def getDatasetName_forPaper(datasetName):
    if datasetName == "Friedman_n100":
        return "F100"
    elif datasetName == "Friedman_n400":
        return "F400"
    elif datasetName == "bodyfat":
        return "body"
    elif datasetName == "housing":
        return "house"
    elif datasetName == "cadata":
        return "cadata"
    elif datasetName == "spacega":
        return "spacega"
    elif datasetName == "syntheticSimpleSin":
        return "bow"
    else:
        assert(False)
        