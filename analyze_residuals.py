
import numpy
import torch
from commonSettings import EstimationType
from commonSettings import ALL_RESULTS_FOLDER
from commonSettings import ALL_FDR_ALPHAS
import commonSettings
import evaluation

import simDataGeneration

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']


# datasetName = "dengue_iquitos"
datasetName = "dengue_san_juan"
# datasetName = "bodyfat"  # all data for fold 0
# datasetName = "cadata" # all data for fold 0
# datasetName = "housing"  # all data for fold 0
# datasetName = "Friedman_n100"  # all data for fold 0
# datasetName = "Friedman_n400" # all data for fold 0
# datasetName = "syntheticSimpleSin" # missing

# NOISE_TYPE = "uniform"
# NOISE_TYPE = "focused"
# NOISE_TYPE = "asymmetric"
NOISE_TYPE = "noNoise"

method = "vanilla"
# method = "student"
# method = "trimmed"
# method = "gamma"
# method = "trimmed_residual_nu"

DATA_SPLIT = "trainTestData"
# DATA_SPLIT = "wholeData"

ALL_RESULTS_FOLDER = "all_results/"
OPTIMIZATION_MODE = "projectedGradient"
PRE_SPECIFIED_NU = 0.5

THRESHOLD = 2.0

NAME_PREFIX = ALL_RESULTS_FOLDER + datasetName + "_" + DATA_SPLIT + "_" + NOISE_TYPE + commonSettings.getNoisePostFix(NOISE_TYPE) + "_" + commonSettings.getMethodPostFix(method, OPTIMIZATION_MODE, PRE_SPECIFIED_NU)


allData = numpy.load(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + DATA_SPLIT + "_" + NOISE_TYPE + commonSettings.getNoisePostFix(NOISE_TYPE) + ".npy", allow_pickle = True).item()

foldId = 0

trueOutlierIndicesZeroOne = allData["all_trueOutlierIndicesZeroOne"][foldId]

allStandardizedResiduals = numpy.load(NAME_PREFIX + "_" + "allStandardizedResiduals.npy", allow_pickle=True)[foldId]

SIGMA_ESTIMATION_TYPES = commonSettings.getSigmaEstimationTypes()
assert(len(SIGMA_ESTIMATION_TYPES) == 1)
sigmaEstimateType = SIGMA_ESTIMATION_TYPES[0]


standardized_residuals = allStandardizedResiduals[sigmaEstimateType]


# differences_in_increasing_residuals = numpy.diff(numpy.sort(standardized_residuals))
# largestDiffId = numpy.argmax(differences_in_increasing_residuals)
# print("largestDiffId = ", largestDiffId)

FULL_N = trueOutlierIndicesZeroOne.shape[0]

print("FULL_N = ", FULL_N)
auc, outlierRecall_topNrTrueOutliers, first_outlier_quantile = evaluation.showOutlierDetectionPerformance_auc_top(trueOutlierIndicesZeroOne, - standardized_residuals)
print("auc = ", auc)
print("outlierRecall_topNrTrueOutliers = ", outlierRecall_topNrTrueOutliers)

print("standardized_residuals = ", standardized_residuals)

estimatedNrOutliers = torch.sum(standardized_residuals > THRESHOLD)
estimatedNrOutliers = estimatedNrOutliers.item()
estimatedOutlierRatio = round((estimatedNrOutliers / FULL_N) * 100.0, 2)
print("estimatedNrOutliers = ", estimatedNrOutliers)
print("estimatedOutlierRatio (in percentage) = ", estimatedOutlierRatio)

fig, ax = plt.subplots()
x = numpy.arange(0, FULL_N, 1)
ax.plot(x, numpy.sort(standardized_residuals), linewidth=2.0, marker = "o")

ax.axhline(y = 2, color = 'r', linestyle = 'dashed') 
ax.axhline(y = 3, color = 'g', linestyle = 'dashed') 

# plt.show()
filename = "all_plots_new/" + "mAnalysis_" + method + "_" + datasetName + "_" + NOISE_TYPE + ".pdf"
plt.savefig(filename)
print("saved to ", filename)

if datasetName.startswith("dengue"):
    assert(foldId == 0)
    X_train = allData["all_X_train"][foldId]
    y_train = allData["all_y_train"][foldId]
    trueOutlierIndicesZeroOne = allData["all_trueOutlierIndicesZeroOne"][foldId]
    
    dataScalerX = allData["all_dataScalerX"][foldId]
    dataScalerY = allData["all_dataScalerY"][foldId]

    X_original = simDataGeneration.transformBack(dataScalerX, X_train)
    y_original = simDataGeneration.transformBack(dataScalerY, y_train) 
    y_original = numpy.square(y_original)


    # print("X_train = ", X_train)
    print("X_original = ", X_original[:,0])
    print("y_original = ", y_original)

    fig, ax = plt.subplots()
    time = numpy.arange(start = 2, stop = y_original.shape[0] + 2)
    ax.plot(time, y_original, linewidth=2.0)
    ax.plot(time[standardized_residuals > THRESHOLD], y_original[standardized_residuals > THRESHOLD], 'ro')
    ax.set_xlabel("days")
    ax.set_ylabel("dengue cases")
    plt.show()
