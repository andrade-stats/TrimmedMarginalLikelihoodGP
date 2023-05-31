
import numpy
import commonSettings
import scipy.stats
import torch

import evaluation

def getNrOutliersEstimate(standardized_abs_residuals, threshold_for_outlier_classification):
    if type(standardized_abs_residuals) is torch.Tensor:
        standardized_abs_residuals = standardized_abs_residuals.numpy()  
    assert(numpy.all(standardized_abs_residuals >= 0)) # we actually saved  |r|

    n = standardized_abs_residuals.shape[0]
    classified_as_outlier_ratio = numpy.sum(standardized_abs_residuals > threshold_for_outlier_classification) / n

    return classified_as_outlier_ratio

def getSimpleDeviationRanking(allData, foldId):
    y_train = allData["all_y_train"][foldId]
    y_train = commonSettings.getTorchTensor(y_train)
    return torch.square(y_train).cpu().numpy()


# caculates R-Precision and AUC scores
def getRankingPerformance(datasetName, noise_type, method, threshold_for_outlier_classification, optimization_mode, pre_specified_nu):
    ALL_RESULTS_FOLDER = "all_results/"
    
    if datasetName.startswith("Friedman") or datasetName.startswith("syntheticSimpleSin"):
        data_split = "trainTestData"
    else:
        data_split = "wholeData"
    
    allData = numpy.load(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + data_split + "_" + noise_type + commonSettings.getNoisePostFix(noise_type) + ".npy", allow_pickle = True).item()
    print(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + data_split + "_" + noise_type + commonSettings.getNoisePostFix(noise_type) + ".npy")

    if method != "simpleDeviationRanking":
        name_prefix = getNamePrefix(datasetName, data_split, noise_type, method, optimization_mode, pre_specified_nu)
        print("name_prefix = ", name_prefix)
        all_allStandardizedResiduals = numpy.load(name_prefix + "_" + "allStandardizedResiduals.npy", allow_pickle=True)
        
    SIGMA_ESTIMATION_TYPES = commonSettings.getSigmaEstimationTypes()
    assert(len(SIGMA_ESTIMATION_TYPES) == 1)

    allAUCs = numpy.zeros(commonSettings.getNrFolds(datasetName, noise_type, data_split))
    allOutlierRecalls = numpy.zeros(commonSettings.getNrFolds(datasetName, noise_type, data_split))
    allFirstOutlierQuantile = numpy.zeros(commonSettings.getNrFolds(datasetName, noise_type, data_split))

    allEstimatedNrOutliers = numpy.zeros(commonSettings.getNrFolds(datasetName, noise_type, data_split))

    for foldId in range(commonSettings.getNrFolds(datasetName, noise_type, data_split)):
        
        if method == "simpleDeviationRanking":
            standardized_residuals = getSimpleDeviationRanking(allData, foldId)
        else:
            allStandardizedResiduals = all_allStandardizedResiduals[foldId]
            standardized_residuals = allStandardizedResiduals[SIGMA_ESTIMATION_TYPES[0]]

        trueOutlierIndicesZeroOne = allData["all_trueOutlierIndicesZeroOne"][foldId]
        allEstimatedNrOutliers[foldId] = getNrOutliersEstimate(standardized_residuals, threshold_for_outlier_classification)
        allAUCs[foldId], allOutlierRecalls[foldId], allFirstOutlierQuantile[foldId] = evaluation.showOutlierDetectionPerformance_auc_top(trueOutlierIndicesZeroOne, - standardized_residuals)
    
    print("allAUCs = ", allAUCs)
    print("allOutlierRecalls = ", allOutlierRecalls)
    
    return allAUCs, allOutlierRecalls, allFirstOutlierQuantile, allEstimatedNrOutliers



def getRegressionPerformance(datasetName, noise_type, method, optimization_mode, pre_specified_nu):
    data_split = "trainTestData"

    name_prefix = getNamePrefix(datasetName, data_split, noise_type, method, optimization_mode, pre_specified_nu)
    
    all_NLL = numpy.load(name_prefix + "_" + "all_NLL.npy", allow_pickle=True)
    all_MSLL = numpy.load(name_prefix + "_" + "all_MSLL.npy", allow_pickle=True)
    all_RMSE = numpy.load(name_prefix + "_" + "all_RMSE.npy", allow_pickle=True)
    all_MedianAbsoluteError = numpy.load(name_prefix + "_" + "all_MediaAbsoluteError.npy", allow_pickle=True)

    return all_NLL, all_MSLL, all_RMSE, all_MedianAbsoluteError


def getNamePrefix(datasetName, data_split, noise_type, method, optimization_mode, pre_specified_nu):
    if data_split is None:
        if datasetName.startswith("Friedman") or datasetName.startswith("syntheticSimpleSin"):
            data_split = "trainTestData"
        else:
            data_split = "wholeData"
    
    return ALL_RESULTS_FOLDER + datasetName + "_" + data_split + "_" + noise_type + commonSettings.getNoisePostFix(noise_type) + "_" + commonSettings.getMethodPostFix(method, optimization_mode, pre_specified_nu)

def getRuntime(datasetName, noise_type, method,  optimization_mode, pre_specified_nu):
    
    allRuntimes = numpy.load(getNamePrefix(datasetName, None, noise_type, method, optimization_mode, pre_specified_nu) + "_" +  "allRuntimes.npy", allow_pickle=True)
    return allRuntimes



def getResultMatrix(allCollectedResults, datasetName, method_with_configs):
    key = datasetName + "$" + method_with_configs
    if key not in allCollectedResults:
        allCollectedResults[key] = numpy.zeros((len(ALL_NOISE_TYPES), commonSettings.GLOBAL_NUMBER_OF_FOLDS)) * numpy.nan

    return allCollectedResults[key]



def createTable(summary, threshold_for_outlier_classification):

    allCollectedResults = {}

    for noise_type_id, noise_type in enumerate(ALL_NOISE_TYPES):
        for datasetName in ALL_DATA_SETS:
            for method_with_configs in ALL_METHODS:
                
                resultMatrix = getResultMatrix(allCollectedResults, datasetName, method_with_configs)

                if method_with_configs.startswith("trimmed_residual_nu"):
                    method = method_with_configs.split("|")[0]
                    optimization_mode = method_with_configs.split("|")[1]
                    pre_specified_nu = 0.5
                elif method_with_configs.startswith("trimmed_informative_nu"):
                    method = method_with_configs.split("|")[0]
                    optimization_mode = method_with_configs.split("|")[1]
                    pre_specified_nu = float(method_with_configs.split("|")[2])
                else:
                    method = method_with_configs
                    optimization_mode = None
                    pre_specified_nu = None

                try:
                    
                    if SHOW_TYPE in ["AUC", "R-PRECISION", "NR_OUTLIERS"]:
                        auc, rPrecision, firstOutlierQuantile, nrOutliersEstimate = getRankingPerformance(datasetName, noise_type, method, threshold_for_outlier_classification, optimization_mode, pre_specified_nu)
                    
                    if SHOW_TYPE == "AUC":
                        assert(noise_type != "noNoise")
                        bestIsHigh = True
                        current_result = auc
                    elif SHOW_TYPE == "R-PRECISION":
                        assert(noise_type != "noNoise")
                        bestIsHigh = True
                        current_result = rPrecision
                    elif SHOW_TYPE == "NR_OUTLIERS":
                        bestIsHigh = None
                        current_result = nrOutliersEstimate
                    elif SHOW_TYPE == "REGRESSION":
                        bestIsHigh = False
                        all_NLL, all_MSLL, all_RMSE, all_MedianAbsoluteError = getRegressionPerformance(datasetName, noise_type, method, optimization_mode, pre_specified_nu)
                        current_result = all_RMSE
                    elif SHOW_TYPE == "MARGINAL_LIKELIHOOD":
                        bestIsHigh = True
                        # average marginal likelihood on trimmed training data
                        current_result = numpy.load(getNamePrefix(datasetName, None, noise_type, method, optimization_mode, pre_specified_nu) + "_" + "allAverageMLL.npy")
                    elif SHOW_TYPE == "RUNTIME":
                        bestIsHigh = False
                        # runtime in minutes
                        current_result = getRuntime(datasetName, noise_type, method, optimization_mode, pre_specified_nu)
                    else:
                        assert(False)

                except FileNotFoundError:
                    current_result = numpy.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * numpy.nan
                    
                resultMatrix[noise_type_id, :] = current_result

    tableStr = ""

    if summary:
        assert(SHOW_TYPE != "NR_OUTLIERS")
        for datasetName in ALL_DATA_SETS:
            allResult_pairs = []
            for method_with_configs in ALL_METHODS:
                resultMatrix = getResultMatrix(allCollectedResults, datasetName, method_with_configs)
                allResult_pairs.append(evaluation.showAvgAndStd(resultMatrix.flatten()))

            tableStr += commonSettings.getDatasetName_forPaper(datasetName) + " & " + evaluation.getHighlightedResults(allResult_pairs, bestIsHigh)  + " \\\\" + "\n" 
    else:
        assert(SHOW_TYPE == "NR_OUTLIERS")
        assert(len(ALL_METHODS) == 1)
        method_with_configs = ALL_METHODS[0]
        for datasetName in ALL_DATA_SETS:
            resultMatrix = getResultMatrix(allCollectedResults, datasetName, method_with_configs)
            allResultStrs = []
            for noise_type_id, _ in enumerate(ALL_NOISE_TYPES):
                allResultStrs.append(evaluation.showAvgAndStd_str(resultMatrix[noise_type_id, :]))

            tableStr += commonSettings.getDatasetName_forPaper(datasetName) + " & " + " & ".join(allResultStrs) + " \\\\" + "\n" 

    return tableStr



ANALYSIS = True

SHOW_TYPE = "R-PRECISION"
# SHOW_TYPE = "NR_OUTLIERS"
# SHOW_TYPE = "REGRESSION"
# SHOW_TYPE = "RUNTIME"
# SHOW_TYPE = "MARGINAL_LIKELIHOOD"

# ALL_NOISE_TYPES = ["noNoise"]
# ALL_NOISE_TYPES = ["uniform"]
ALL_NOISE_TYPES = ["focused"]
# ALL_NOISE_TYPES = ["asymmetric"]

# ALL_NOISE_TYPES = ["noNoise", "uniform", "focused", "asymmetric"]
# ALL_NOISE_TYPES = ["uniform", "focused", "asymmetric"]

commonSettings.TRUE_OUTLIER_RATIO_FOR_NOISE = 0.1


if ANALYSIS:
    # ALL_DATA_SETS = ["syntheticSimpleSin", "Friedman_n100", "Friedman_n400", "bodyfat", "housing", "spacega"]
    # ALL_METHODS = ["trimmed_informative_nu_withoutCV|projectedGradient|0.2", "trimmed_informative_nu_withoutCV|greedy_RemoveBatch|0.2", "trimmed_informative_nu_withoutCV|greedy_RemoveOneByOne|0.2"]
    # ALL_METHODS = ["trimmed_residual_nu|projectedGradient"]
    ALL_DATA_SETS = ["Friedman_n100"]
    ALL_METHODS = ["student"] + ["trimmed_informative_nu_withoutCV|projectedGradient|0.2"]
else:
    ALL_DATA_SETS = ["syntheticSimpleSin", "Friedman_n100", "Friedman_n400", "bodyfat", "housing", "spacega"]
    # ALL_METHODS = commonSettings.ALL_BASELINE_METHODS + ["trimmed_residual_nu|projectedGradient"]
    ALL_METHODS = commonSettings.ALL_BASELINE_METHODS + ["trimmed_informative_nu|projectedGradient|0.5"]
    

ALL_RESULTS_FOLDER = "all_results/"

ALPHA = 0.01
threshold_for_outlier_classification = scipy.stats.norm.ppf(1.0 - ALPHA, loc=0, scale=1)
# TRUE_OUTLIER_RATIO = commonSettings.getTrueOutlierRatio(NOISE_TYPE)
# ideal_classified_as_outlier_ratio = TRUE_OUTLIER_RATIO + 2.0 * ALPHA * (1.0 - TRUE_OUTLIER_RATIO)

tableStr = createTable(SHOW_TYPE != "NR_OUTLIERS", threshold_for_outlier_classification)


print("********************************************")
print("TRUE_OUTLIER_RATIO_FOR_NOISE = ", commonSettings.TRUE_OUTLIER_RATIO_FOR_NOISE)
print("********************************************")
print("***** " + SHOW_TYPE + " (" + str(ALL_NOISE_TYPES) + ") *********************")
print("********************************************")
print(tableStr)
