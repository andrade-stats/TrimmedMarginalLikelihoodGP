
import sklearn.datasets
import sklearn.preprocessing

import copy
import numpy
import sklearn.linear_model
import sklearn.metrics
import time
import sys

import random
import torch

from commonSettings import EstimationType
from commonSettings import ALL_RESULTS_FOLDER
import commonSettings

import simDataGeneration

import evaluation

import commons_GP
import trimmedGP
import gpytorch

# print("GPyTorch - Version = ", gpytorch.__version__)
# print("PyTorch - Version = ", torch.__version__)


if len(sys.argv) == 1:
    
    VISUALIZE = False

    # datasetName ="TopGear"
    # datasetName = "dengue_iquitos"
    # datasetName = "dengue_san_juan"
    # datasetName = "bodyfat"
    # datasetName = "cadata"
    # datasetName = "housing"
    # datasetName = "syntheticSimpleSin"
    # datasetName = "spacega"
    datasetName = "Friedman_n100"
    # datasetName = "Friedman_n400"
    # datasetName = "syntheticFewDotData"

    # datasetName = "syntheticLinearCorrelated"
    
    # modelName = "linearRegression"
    # modelName = "NN"
    
    # METHOD = "standardNoOutlier"
    # METHOD = "HardSort"
    # METHOD = "PostProcessing"
    # METHOD = "HardSort"
    # METHOD = "SoftSort"
    # METHOD = "fdrProposed"
    # METHOD = "fdrBaseline"
    # METHOD = "BetaDivergence"
    # METHOD = "GammaDivergence"
    # BETA = 0.3
    # BETA = 0.0
    
    # method = "vanilla"
    # method = "gamma"
    # method = "student"
    
    method = "trimmed_informative_nu_withoutCV"
    # method = "trimmed_residual_nu"
    # method = "trimmed_informative_nu"
    
    # ALL_NOISE_TYPES = ["noNoise", "focused", "uniform", "asymmetric"]
    ALL_NOISE_TYPES = ["focused"]
    # ALL_NOISE_TYPES = ["uniform"]
    # ALL_NOISE_TYPES = ["noNoise"]
    # ALL_NOISE_TYPES = ["asymmetric"]
    
    
    OPTIMIZATION_MODE = "projectedGradient"
    # OPTIMIZATION_MODE = "greedy_RemoveOneByOne"
    # OPTIMIZATION_MODE = "greedy_RemoveBatch"
    # OPTIMIZATION_MODE = "LMDbaseline"
    
    commonSettings.TRUE_OUTLIER_RATIO_FOR_NOISE = 0.1

    DATA_SPLIT = "trainTestData"
    # DATA_SPLIT = "wholeData"
    

    if method.startswith("trimmed_informative_nu"):
        PRE_SPECIFIED_NU = 0.2
    else:
        PRE_SPECIFIED_NU = 0.5

else:
    
    VISUALIZE = False

    assert(len(sys.argv) >= 4)
    datasetName_withDataSplitInfo = sys.argv[1]
    
    parts = datasetName_withDataSplitInfo.split("_")

    if datasetName_withDataSplitInfo.startswith("dengue"):
        datasetName = datasetName_withDataSplitInfo
        DATA_SPLIT = "trainTestData"
    elif datasetName_withDataSplitInfo.startswith("housing") or datasetName_withDataSplitInfo.startswith("bodyfat") or datasetName_withDataSplitInfo.startswith("spacega"):
        assert(len(parts) == 2)
        datasetName = parts[0]
        DATA_SPLIT = parts[1]
        assert(DATA_SPLIT == "trainTestData" or DATA_SPLIT == "wholeData")
    else:
        assert(len(parts) == 1 or parts[1].startswith("n"))
        DATA_SPLIT = "trainTestData"
        datasetName = datasetName_withDataSplitInfo

    method = sys.argv[2]
    ALL_NOISE_TYPES_CSV = sys.argv[3]
    ALL_NOISE_TYPES = ALL_NOISE_TYPES_CSV.split(",")

    if method.startswith("trimmed"):
        OPTIMIZATION_MODE = sys.argv[4]
    else:
        assert(sys.argv[4] == "None")
        OPTIMIZATION_MODE = None
    
    commonSettings.TRUE_OUTLIER_RATIO_FOR_NOISE = float(sys.argv[5])
    
    if method.startswith("trimmed_informative_nu"):
        PRE_SPECIFIED_NU = float(sys.argv[6])
    else:
        PRE_SPECIFIED_NU = 0.5
        

assert(commonSettings.TRUE_OUTLIER_RATIO_FOR_NOISE >= 0.0 and commonSettings.TRUE_OUTLIER_RATIO_FOR_NOISE <= 0.4)
assert(PRE_SPECIFIED_NU <= 0.5 and PRE_SPECIFIED_NU > 0.0)
assert(commonSettings.TRUE_OUTLIER_RATIO_FOR_NOISE <= PRE_SPECIFIED_NU)

if method.startswith("trimmed"):
    assert(OPTIMIZATION_MODE == "projectedGradient" or OPTIMIZATION_MODE.startswith("greedy") or OPTIMIZATION_MODE == "LMDbaseline")

print("ALL_NOISE_TYPES = ", ALL_NOISE_TYPES)

commonSettings.setDevice()

for NOISE_TYPE in ALL_NOISE_TYPES:

    NUMBER_OF_FOLDS = commonSettings.getNrFolds(datasetName, NOISE_TYPE, DATA_SPLIT)
        
    numpy.random.seed(3523421)
    torch.manual_seed(3523421)
    random.seed(3523421)
    
    allData = numpy.load(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + DATA_SPLIT + "_" + NOISE_TYPE + commonSettings.getNoisePostFix(NOISE_TYPE) + ".npy", allow_pickle = True).item()


    SIGMA_ESTIMATION_TYPES = commonSettings.getSigmaEstimationTypes()

    all_estimatedNoiseVariance_correctedForScale = {}
    
    for sigmaEstimationType in SIGMA_ESTIMATION_TYPES:
        all_estimatedNoiseVariance_correctedForScale[sigmaEstimationType] = numpy.zeros(NUMBER_OF_FOLDS)
    
    
    NR_OUTLIER_UPPER_BOUND = int((allData["all_y_train"][0]).shape[0] * PRE_SPECIFIED_NU)

    
    print("TOTAL NUMBER OF SAMPLES = ", (allData["all_y_train"][0]).shape[0])
    print("PRE_SPECIFIED_NU = ", PRE_SPECIFIED_NU)
    print("NR_OUTLIER_UPPER_BOUND = ", NR_OUTLIER_UPPER_BOUND)
    
    if datasetName == "syntheticFewDotData":
        assert(NR_OUTLIER_UPPER_BOUND == 1)

    
    all_allPValues_logScale = numpy.empty(NUMBER_OF_FOLDS, dtype=object)
    allRuntimes = numpy.zeros(NUMBER_OF_FOLDS)

    allFinalNuEstimates = numpy.zeros(NUMBER_OF_FOLDS)
    all_mean_predictions_test = numpy.empty(NUMBER_OF_FOLDS, dtype=object)
    allStandardizedResiduals = numpy.empty(NUMBER_OF_FOLDS, dtype=object)

    all_allEstimatedNoiseVariances = numpy.empty(NUMBER_OF_FOLDS, dtype=object)
    all_GPmodels = numpy.empty(NUMBER_OF_FOLDS, dtype=object)
    startTime = time.time()

    all_NLL = numpy.zeros(NUMBER_OF_FOLDS)
    all_MSLL = numpy.zeros(NUMBER_OF_FOLDS)
    all_RMSE = numpy.zeros(NUMBER_OF_FOLDS)
    all_MedianAbsoluteError = numpy.zeros(NUMBER_OF_FOLDS)

    allAverageMLL = numpy.zeros(NUMBER_OF_FOLDS) * numpy.nan

    for foldId in range(NUMBER_OF_FOLDS):
        
        print(f"********************** data fold id = {foldId} **********************")
        
        X_train = allData["all_X_train"][foldId]
        y_train = allData["all_y_train"][foldId]
        trueOutlierIndicesZeroOne = allData["all_trueOutlierIndicesZeroOne"][foldId]
        dataScalerX = allData["all_dataScalerX"][foldId]
        dataScalerY = allData["all_dataScalerY"][foldId]

        if dataScalerY is not None:
            yScaling = dataScalerY.scale_[0]
        else:
            yScaling = 1.0
        
        
        X_train = commonSettings.getTorchTensor(X_train)
        y_train = commonSettings.getTorchTensor(y_train)
        

        if datasetName == "housing": 
            # for housing GP is not numerically stable with default setting of jitter 
            JITTER_FOR_GPY_TORCH = 1e-1
        else:
            JITTER_FOR_GPY_TORCH = 1e-6

        with gpytorch.settings.cholesky_jitter(float_value = JITTER_FOR_GPY_TORCH, double_value = JITTER_FOR_GPY_TORCH), gpytorch.settings.cholesky_max_tries(10), gpytorch.settings.variational_cholesky_jitter(float_value = JITTER_FOR_GPY_TORCH, double_value = JITTER_FOR_GPY_TORCH): 
            
            startTime_gp_training = time.time()

            if method == "vanilla":
                allPValues_logScale, allEstimatedNoiseVariances, gpModel, average_marginalLikelihood, standardizedResiduals = commons_GP.trainVanillaGP(X_train, y_train, sigmaEstimateTypes = SIGMA_ESTIMATION_TYPES)
            elif method == "gamma":
                allPValues_logScale, allEstimatedNoiseVariances, gpModel, average_marginalLikelihood, standardizedResiduals = commons_GP.trainGP_withGammaDivergence(X_train, y_train, sigmaEstimateTypes = SIGMA_ESTIMATION_TYPES)
            elif method == "student":
                allPValues_logScale, allEstimatedNoiseVariances, gpModel, average_marginalLikelihood, standardizedResiduals = commons_GP.trainStudentTGP(X_train, y_train, sigmaEstimateTypes = SIGMA_ESTIMATION_TYPES)
            
            elif method == "trimmed_residual_nu":

                # proposed method with automatic selection of nu (Algorithm 2 in UAI paper)

                estimated_maxNrOutlierSamples = NR_OUTLIER_UPPER_BOUND
                all_estimated_nus = [estimated_maxNrOutlierSamples / X_train.shape[0]]
                while(True):
                    estimated_maxNrOutlierSamples, allPValues_logScale, allEstimatedNoiseVariances, standardizedResiduals = trimmedGP.residualNuTrimmedGP(X_train, y_train, maxNrOutlierSamples = estimated_maxNrOutlierSamples, method = OPTIMIZATION_MODE)
                    all_estimated_nus.append(estimated_maxNrOutlierSamples / X_train.shape[0])
                    if all_estimated_nus[-2] <= all_estimated_nus[-1]:
                        gpModel = trimmedGP.getFinalModelForPrediction(X_train, y_train, estimated_maxNrOutlierSamples, allEstimatedNoiseVariances, method = OPTIMIZATION_MODE)
                        break
                
                allFinalNuEstimates[foldId] = all_estimated_nus[-1]
                print("all_estimated_nus = ", all_estimated_nus)
                print("final_nu_estimate = ", allFinalNuEstimates[foldId])
                if all_estimated_nus[-2] < all_estimated_nus[-1]:
                    print("SEEMS OPTIMIZATION WAS NOT OPTIMAL")
                    

            elif method == "trimmed_informative_nu":

                # proposed method with fixed nu, whereas variance is estimated using cross-validation

                _, allPValues_logScale, allEstimatedNoiseVariances, standardizedResiduals = trimmedGP.residualNuTrimmedGP(X_train, y_train, maxNrOutlierSamples = NR_OUTLIER_UPPER_BOUND, method = OPTIMIZATION_MODE)
                gpModel = trimmedGP.getFinalModelForPrediction(X_train, y_train, NR_OUTLIER_UPPER_BOUND, allEstimatedNoiseVariances, method = OPTIMIZATION_MODE)
                allAverageMLL[foldId] = gpModel.getMLL()

            elif method == "trimmed_informative_nu_withoutCV":
                
                # proposed method with fixed nu, whereas variance is estimated without cross-validation

                if OPTIMIZATION_MODE == "LMDbaseline":
                    _, _, allPValues_logScale, allEstimatedNoiseVariances, gpModel, standardizedResiduals = commons_GP.LMD(X_train, y_train, maxNrOutlierSamples = NR_OUTLIER_UPPER_BOUND)
                else:
                    allPValues_logScale, allEstimatedNoiseVariances, gpModel, _, _ = trimmedGP.trainTrimmedGP(X_train, y_train, maxNrOutlierSamples = NR_OUTLIER_UPPER_BOUND, sigmaEstimateTypes = commonSettings.getSigmaEstimationTypes(), optimizationMode = OPTIMIZATION_MODE)
                    standardizedResiduals = commons_GP.getAllRobustEstimates(gpModel, X_train, y_train)
                
                allAverageMLL[foldId] = gpModel.getMLL()

            else:
                assert(False)
        
            allStandardizedResiduals[foldId] = standardizedResiduals
            allRuntimes[foldId] = (time.time() - startTime_gp_training) / 60.0         

            if DATA_SPLIT == "trainTestData":
                
                X_cleanTest = allData["all_X_cleanTest"][foldId]
                y_cleanTest = allData["all_y_cleanTest"][foldId]
                
                X_cleanTest = commonSettings.getTorchTensor(X_cleanTest)
                y_cleanTest = commonSettings.getTorchTensor(y_cleanTest)
                
                all_NLL[foldId], all_MSLL[foldId], all_RMSE[foldId], all_MedianAbsoluteError[foldId] = gpModel.evaluatePredictions(X_cleanTest, y_cleanTest)
        

        all_allEstimatedNoiseVariances[foldId] = allEstimatedNoiseVariances
        all_allPValues_logScale[foldId] = allPValues_logScale

        if VISUALIZE and (datasetName == "syntheticSimpleSin" or datasetName.startswith("syntheticFewDotData")):
            
            simDataGeneration.saveDataForPlot(datasetName, NOISE_TYPE, DATA_SPLIT, foldId, method, OPTIMIZATION_MODE, PRE_SPECIFIED_NU, X_train, dataScalerX, dataScalerY, gpModel)
            
            # simDataGeneration.visualizePredictions(datasetName,  X_train, y_train, trueOutlierIndicesZeroOne, dataScalerX, dataScalerY, gpModel)
            # simDataGeneration.showBowShapedData(datasetName, NOISE_TYPE, X_train, y_train, dataScalerX, dataScalerY)
            break

        print("********************************************")

    if VISUALIZE and (datasetName == "syntheticSimpleSin" or datasetName.startswith("syntheticFewDotData")):
        continue

    
    assert(not VISUALIZE)

    NAME_PREFIX = ALL_RESULTS_FOLDER + datasetName + "_" + DATA_SPLIT + "_" + NOISE_TYPE + commonSettings.getNoisePostFix(NOISE_TYPE) + "_" + commonSettings.getMethodPostFix(method, OPTIMIZATION_MODE, PRE_SPECIFIED_NU)
    
    numpy.save(NAME_PREFIX + "_" + "all_allEstimatedNoiseVariances",  all_allEstimatedNoiseVariances)
    
    numpy.save(NAME_PREFIX + "_" + "allStandardizedResiduals",  allStandardizedResiduals)
    numpy.save(NAME_PREFIX  + "_" + "all_allPValues_logScale",  all_allPValues_logScale)
    numpy.save(NAME_PREFIX + "_" + "allAverageMLL", allAverageMLL)

    numpy.save(NAME_PREFIX + "_" +  "allRuntimes",  allRuntimes)

    if DATA_SPLIT == "trainTestData":
        numpy.save(NAME_PREFIX + "_" + "all_NLL",  all_NLL)
        numpy.save(NAME_PREFIX + "_" + "all_MSLL",  all_MSLL)
        numpy.save(NAME_PREFIX + "_" + "all_RMSE",  all_RMSE)
        numpy.save(NAME_PREFIX + "_" + "all_MediaAbsoluteError",  all_MedianAbsoluteError)
        
        print("******************************")
        print("NLL = ", evaluation.showAvgAndStd_str(all_NLL))
        print("MSLL = ", evaluation.showAvgAndStd_str(all_MSLL))
        print("RMSE = ", evaluation.showAvgAndStd_str(all_RMSE))
        print("MedianAbsoluteError = ", evaluation.showAvgAndStd(all_MedianAbsoluteError))
        print("******************************")

    if method == "trimmed_residual_nu":
        numpy.save(NAME_PREFIX  + "_" + "allFinalNuEstimates", allFinalNuEstimates)
    

    print("******************************")
    print("FINISHED")
    print("RUNTIME (in minutes) = ", evaluation.showAvgAndStd(allRuntimes))
    print("commonSettings.GLOBAL_NUMBER_OF_FOLDS = ", commonSettings.GLOBAL_NUMBER_OF_FOLDS)
    print("total runtime (in minutes) = ", (time.time() - startTime) / 60.0)