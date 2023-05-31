import numpy
import commonSettings

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

matplotlib.use('tkagg')

import simDataGeneration

def showOnlyData(plt, datasetName, trueOutlierIndicesZeroOne):
    plt.tight_layout()

    lw = 2
    plt.scatter(X[trueOutlierIndicesZeroOne == 0,0], y[trueOutlierIndicesZeroOne == 0], s = 5, c='b', label='inlier')
    plt.scatter(X[trueOutlierIndicesZeroOne == 1,0], y[trueOutlierIndicesZeroOne == 1], s = 5, c='m', label='outlier')
    plt.ylim([-3.5, 3.5])
    plt.yticks(Y_AXIS_TICKS)
    plt.xticks(X_AXIS_TICKS)


    x_plot = simDataGeneration.getScaledXData_forSyntheticSimpleSin(numpy.linspace(0, 1.0, 10000))
    trueY = simDataGeneration.getResponse_forSyntheticSimpleSin(x_plot)
    plt.plot(x_plot, trueY, color='navy', lw=lw, label='True')

    plt.savefig("all_plots/" + datasetName + "_" + NOISE_TYPE + "_onlyData" + ".pdf")

    fig = plt.figure()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    return

def showAllLearnedFunctions(plt, datasetName):
    # plt.figure(figsize=(10, 15))
    plt.tight_layout()
    
    plt.suptitle(NOISE_TYPE + " " + "outliers")

    allSubPlots = []
    allSubPlots.append(plt.subplot2grid((2, 2), (0, 0), colspan=1))
    allSubPlots.append(plt.subplot2grid((2, 2), (0, 1), colspan=1))
    allSubPlots.append(plt.subplot2grid((2, 2), (1, 0), colspan=1))
    allSubPlots.append(plt.subplot2grid((2, 2), (1, 1), colspan=1))

    for methodId in range(len(ALL_METHOD_NAMES)):
        method = ALL_METHOD_NAMES[methodId]

        lw = 2
        allSubPlots[methodId].scatter(X[trueOutlierIndicesZeroOne == 0,0], y[trueOutlierIndicesZeroOne == 0], s = 5, c='b', label='inlier')
        allSubPlots[methodId].scatter(X[trueOutlierIndicesZeroOne == 1,0], y[trueOutlierIndicesZeroOne == 1], s = 5, c='m', label='outlier')

        x_plot = simDataGeneration.getScaledXData_forSyntheticSimpleSin(numpy.linspace(0, 1.0, 10000))
        trueY = simDataGeneration.getResponse_forSyntheticSimpleSin(x_plot)
        allSubPlots[methodId].plot(x_plot, trueY, color='navy', lw=lw, label='True')
        
        PLOT_FILENAME_PREFIX =  datasetName + "_" + DATA_SPLIT + "_" + NOISE_TYPE + commonSettings.getNoisePostFix(NOISE_TYPE) + "_" + commonSettings.getMethodPostFix(method, OPTIMIZATION_MODE, PRE_SPECIFIED_NU) 
        allResultsForPlot = numpy.load("all_plots/" + PLOT_FILENAME_PREFIX + "_" + str(foldId) + "_data.npy", allow_pickle = True).item()
        allSubPlots[methodId].plot(allResultsForPlot["predicted_x"], allResultsForPlot["predicted_y"], color='red', lw=lw, label='Estimate')
        
        allSubPlots[methodId].set_ylim([-3.5, 3.5])
        allSubPlots[methodId].set_yticks(Y_AXIS_TICKS)
        allSubPlots[methodId].set_xticks(X_AXIS_TICKS)

        if methodId <= 1:
            allSubPlots[methodId].set_xticklabels([])
        else:
            allSubPlots[methodId].set_xlabel('x')
        
        if methodId == 0 or methodId == 2:
            allSubPlots[methodId].set_ylabel('y')
        
        allSubPlots[methodId].title.set_text(commonSettings.getLabelName(method))

    
    # plt.show()
    plt.savefig("all_plots/" + datasetName + "_" + NOISE_TYPE + "_all" + ".pdf")


datasetName = "syntheticSimpleSin"

NOISE_TYPE = "uniform"
# NOISE_TYPE = "focused"
# NOISE_TYPE = "asymmetric"
# NOISE_TYPE = "noNoise"

# ALL_METHOD_NAMES = ["vanilla", "student", "gamma", "trimmed_informative_nu_withoutCV"]
ALL_METHOD_NAMES = ["vanilla", "student", "gamma", "trimmed_residual_nu"]
# ALL_METHOD_NAMES = ["trimmed_informative_nu"]
# method = "student"
# method = "trimmed"
# method = "gamma"

OPTIMIZATION_MODE = "projectedGradient"
# OPTIMIZATION_MODE = "greedy_RemoveBatch"
# OPTIMIZATION_MODE = "LMDbaseline"

DATA_SPLIT = "trainTestData"
# DATA_SPLIT = "wholeData"

if datasetName == "syntheticSimpleSin":
    PRE_SPECIFIED_NU = 0.2
else:
    PRE_SPECIFIED_NU = 0.5
    
foldId = 0

DATA_SPLIT = "trainTestData"



# X_AXIS_TICKS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
X_AXIS_TICKS = [-1.0, 0.0, 1.0]
Y_AXIS_TICKS = [-3,-2,-1,0,1,2,3]

allData = numpy.load(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + DATA_SPLIT + "_" + NOISE_TYPE + commonSettings.getNoisePostFix(NOISE_TYPE) + ".npy", allow_pickle = True).item()

transformed_X = allData["all_X_train"][foldId]
transformed_y = allData["all_y_train"][foldId]
trueOutlierIndicesZeroOne = allData["all_trueOutlierIndicesZeroOne"][foldId]
dataScalerX = allData["all_dataScalerX"][foldId]
dataScalerY = allData["all_dataScalerY"][foldId]

X = simDataGeneration.transformBack(dataScalerX, transformed_X)
y = simDataGeneration.transformBack(dataScalerY, transformed_y) 

# ******** show only data ********
# showOnlyData(plt,datasetName, trueOutlierIndicesZeroOne)


# ******** show function learned by each method ********
showAllLearnedFunctions(plt, datasetName)



