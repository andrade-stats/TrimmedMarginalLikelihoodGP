
import numpy
import commonSettings

def generateData(responseStd, beta, tau, correlationMatrix, n):
    
    zeroVec = numpy.zeros(beta.shape[0])
    X = numpy.random.multivariate_normal(zeroVec, correlationMatrix, n)
    y = numpy.matmul(X, beta)
    y += tau
    
    noise = responseStd * numpy.random.normal(size=n)
    y += noise
    
    return X, y

    
def addNoise(trueBeta, noiseRatio):
    if noiseRatio > 0.0:
        assert(noiseRatio == 0.5 or noiseRatio == 0.2)
        
        if trueBeta.shape[0] > 10:
            MIN_VALUE = noiseRatio
            p = trueBeta.shape[0]
            contaminatedNumber = int(p * 0.01)
            nextFreePosition = numpy.max(numpy.where(trueBeta != 0)[0]) + 1
            trueBeta[nextFreePosition : (nextFreePosition + contaminatedNumber)] = numpy.random.uniform(low = -MIN_VALUE, high = MIN_VALUE, size = contaminatedNumber) 
        else:
            MIN_VALUE = noiseRatio
            numberOfZeros = numpy.sum(trueBeta == 0)
            trueBeta[trueBeta == 0] = numpy.random.uniform(low = -MIN_VALUE, high = MIN_VALUE, size = numberOfZeros) 
    
    return trueBeta



# generates data as in "Regression Shrinkage and Selection via the Lasso"
def generateLassoExampleData(exampleType, noiseRatio, lowResponseStd):
    
    if exampleType == "example1":
        trueBeta = numpy.asarray([3, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        responseStd = 3.0
    if exampleType == "exampleOneHuge":
        trueBeta = numpy.asarray([1000.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        responseStd = 3.0
    elif exampleType == "example3":
        trueBeta = numpy.asarray([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        responseStd = 2.0
    elif exampleType == "myExample":
        trueBeta = numpy.asarray([3.0, 2.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        responseStd = 1.0
    
    if lowResponseStd:
        responseStd = responseStd * 0.1
    
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    p = trueBeta.shape[0]
    rho = 0.5
    correlationMatrix = numpy.zeros((p,p))
    for i in range(p):
        for j in range(p):
            correlationMatrix[i,j] = rho ** numpy.abs(i-j)
    
    return trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix


def generateOrthogonalDataExample(exampleType, nrTrueSamples, nrOutlierSamples, noiseRatio, lowResponseStd):
    assert(exampleType == "example1")
    
    trueTau = 0.0
    trueBeta = numpy.asarray([3, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    responseStd = 3.0
    
    if lowResponseStd:
        responseStd = responseStd * 0.1
        
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    p = trueBeta.shape[0]
    correlationMatrix = numpy.eye(p)
    # for i in range(p):
    #    for j in range(p):
    #        correlationMatrix[i,j] = rho ** numpy.abs(i-j)
    
    # X, y = generateData(responseStd, trueBeta, trueTau, correlationMatrix, nrTrueSamples, nrOutlierSamples)
    
    return trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix


# generates data as in "Extended Bayesian information criteria for model selection with large model spaces"
def generateEBICExampleData(p, nrTrueSamples, nrOutlierSamples, noiseRatio):
    
    trueTau = 0.0
    responseStd = 1.0
    trueBetaInitial = numpy.asarray([0.0, 1.0, 0.7, 0.5, 0.3, 0.2])
    
    trueBeta = numpy.zeros(p)
    trueBeta[0:trueBetaInitial.shape[0]] = trueBetaInitial
    
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    rho = 0.5
    correlationMatrix = numpy.zeros((p,p))
    for i in range(p):
        for j in range(p):
            correlationMatrix[i,j] = rho ** numpy.abs(i-j)
    
    X, y = generateData(responseStd, trueBeta, trueTau, correlationMatrix, nrTrueSamples, nrOutlierSamples)
    
    return X, y, trueBeta, trueBetaWithoutNoiseOrOutlier


# generates the data as in Section 4 of "The EM Approach to Bayesian Variable Selection" (they use p = 1000 and n = 100)
def generateEMVSExampleData(noiseRatio):
    
    p = 1000
    responseStd = numpy.sqrt(3)
    
    trueBetaInitial = numpy.asarray([3.0,2.0,1.0])
    
    trueBeta = numpy.zeros(p)
    trueBeta[0:trueBetaInitial.shape[0]] = trueBetaInitial
    
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    rho = 0.6
    correlationMatrix = numpy.zeros((p,p))
    for i in range(p):
        for j in range(p):
            correlationMatrix[i,j] = rho ** numpy.abs(i-j)
    
    return trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix


def highDimOrthogonal(nrTrueSamples, nrOutlierSamples, noiseRatio):
    assert(nrTrueSamples >= 100)
    
    trueTau = 0.0
    
    p = 1000
    responseStd = numpy.sqrt(3)
    
    trueBetaInitial = numpy.asarray([3.0,2.0,1.0])
    
    trueBeta = numpy.zeros(p)
    trueBeta[0:trueBetaInitial.shape[0]] = trueBetaInitial
    
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    correlationMatrix = numpy.eye(p)
    X, y = generateData(responseStd, trueBeta, trueTau, correlationMatrix, nrTrueSamples, nrOutlierSamples)
    
    return X, y, trueBeta, trueBetaWithoutNoiseOrOutlier


# checked
def getExpectedMSE(trueBeta, correlationMatrix, responseStd, estimatedBeta):
    mse = responseStd ** 2
    mse += (trueBeta - estimatedBeta) @ correlationMatrix @ (trueBeta - estimatedBeta)
    return mse



def getSyntheticLinearData(dataType, n, nrRepetitions):
    
    RANDOM_GENERATOR_SEED = 9899832
    numpy.random.seed(RANDOM_GENERATOR_SEED)
    
    trueTau = 0.0
    
    lowResponseStd = False
    noiseRatioOnRegressionCoefficients = 0.0
    
    
    allX = []
    allY = []
    
    if dataType == "highDim":
        trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix = generateEMVSExampleData(noiseRatioOnRegressionCoefficients)
    elif dataType == "highDimOr":
        trueBeta, trueBetaWithoutNoiseOrOutlier = highDimOrthogonal(n, 0, noiseRatioOnRegressionCoefficients)
    elif dataType == "correlated":
        trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix  = generateLassoExampleData("example1", noiseRatioOnRegressionCoefficients, lowResponseStd)
    elif dataType == "orthogonal":
        trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix  = generateOrthogonalDataExample("example1", n, 0.0, noiseRatioOnRegressionCoefficients, lowResponseStd)
    elif dataType == "oneHuge":
        trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix  = generateLassoExampleData("exampleOneHuge", noiseRatioOnRegressionCoefficients, lowResponseStd)
    else:
        assert(False)
    
    # print("trueBeta = ", trueBeta)
    # print("responseStd = ", responseStd)
#     print("trueBetaWithoutNoiseOrOutlier = ", trueBetaWithoutNoiseOrOutlier)
#     
#     bestMSE = getExpectedMSE(trueBeta, correlationMatrix, responseStd, trueBeta)
#     simplifiedMSE = getExpectedMSE(trueBeta, correlationMatrix, responseStd, trueBetaWithoutNoiseOrOutlier)
#     mseIncrease = (simplifiedMSE / bestMSE) - 1.0
#     print("expected MSE of true beta = ", bestMSE)
#     print("expected MSE of simplified beta = ", simplifiedMSE)
#     print("expected increase in MSE of simplified beta = ", mseIncrease)
    
    
    
    for repetitionId in range(nrRepetitions):
        X, y = generateData(responseStd, trueBeta, trueTau, correlationMatrix, n)
    
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
    
        allX.append(X)
        allY.append(y)
    
    
    return allX, allY, trueBeta, responseStd, correlationMatrix 


def getScaledXData_forSyntheticSimpleSin(originalX):
    return (originalX - 0.5) * numpy.sqrt(12.0)

def getResponse_forSyntheticSimpleSin(x):
    return 3.235 * (numpy.sin(((x / numpy.sqrt(12.0)) + 0.5) * numpy.pi)) - 2.058

def getSyntheticNonLinearData(dataType, n, nrVariables, nrRepetitions):
    
    RANDOM_GENERATOR_SEED = 9899832
    numpy.random.seed(RANDOM_GENERATOR_SEED)
    
    trueTau = 0.0
    
    responseStd = 0.1
    
    
    allX = []
    allY = []
    
    for repetitionId in range(nrRepetitions):
        if dataType == "syntheticSimpleSin":
            X = getScaledXData_forSyntheticSimpleSin(numpy.random.rand(n, nrVariables))
            trueY = getResponse_forSyntheticSimpleSin(X[:, 0])
        else:
            assert(False)
            
        noise = responseStd * numpy.random.normal(size=n)
        y = trueY + noise + trueTau

        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        print("synthetic data:")
        print("variance(y) = ", numpy.var(y))
        print("mean(y) = ", numpy.mean(y))

        allX.append(X)
        allY.append(y)
    
    # assert(False)
    
    trueNoiseVariance = responseStd ** 2
    return allX, allY, trueNoiseVariance


# Data from J.H. Friedman, Multivariate adaptive regression splines, Ann. Stat. 19 (1) (1991) 1–67.
# exactly as in "Robust Regression with twinned Gaussian Processes", page 7
def getFriedmanData(n, nrRepetitions):
    nrVariables = 10

    RANDOM_GENERATOR_SEED = 9899832
    numpy.random.seed(RANDOM_GENERATOR_SEED)
    
    allX = []
    allY = []
    
    responseStd = 1.0

    for repetitionId in range(nrRepetitions):
        
        X = numpy.random.rand(n, nrVariables)
        trueY = 10.0 * numpy.sin(numpy.pi * X[:,0] * X[:,1]) + 20.0 * numpy.square(X[:,2] - 0.5) + 10.0 * X[:,3] + 5.0 * X[:,4]
        
        noise = responseStd * numpy.random.normal(size=n)
        y = trueY + noise

        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        print("synthetic data:")
        print("variance(y) = ", numpy.var(y))
        print("mean(y) = ", numpy.mean(y))

        allX.append(X)
        allY.append(y)
    
    trueNoiseVariance = responseStd ** 2
    return allX, allY, trueNoiseVariance


def getFewDotData(nrRepetitions):
    X = numpy.asarray([[1.0], [2.0], [3.0], [4.0], [5.0], [10.0]])
    y = numpy.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 1.0])

    allX = []
    allY = []
    
    responseStd = 1.0

    for repetitionId in range(nrRepetitions):
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        allX.append(X)
        allY.append(y)
    
    trueNoiseVariance = responseStd ** 2
    return allX, allY, trueNoiseVariance


def getUniformSampledX_1D_forPlot(transformed_X):
    minVal = numpy.min(transformed_X.numpy()[:,0])
    maxVal = numpy.max(transformed_X.numpy()[:,0])
    sampledX = numpy.linspace(minVal, maxVal, 10000)
    return numpy.reshape(sampledX, (-1, 1))


def transformBack(dataScaler, transformed_data):
    if (len(transformed_data.shape) == 1):
        return dataScaler.inverse_transform(numpy.reshape(transformed_data, (-1,1)))[:,0]
    else:
        return dataScaler.inverse_transform(transformed_data)
        

def saveDataForPlot(datasetName, noiseType, dataSplit, foldId, method, OPTIMIZATION_MODE, PRE_SPECIFIED_NU, transformed_X, dataScalerX, dataScalerY, gpModel):
    assert(datasetName == "syntheticSimpleSin")
    
    allResultsForPlot = {}
        
    sampledX_transformedScale = getUniformSampledX_1D_forPlot(transformed_X)
    sampledX_transformedScale = commonSettings.getTorchTensor(sampledX_transformedScale)
    meanPredictions_transformedScale = gpModel.getMeanPrediction(sampledX_transformedScale)
    
    allResultsForPlot["predicted_x"] = transformBack(dataScalerX, sampledX_transformedScale)
    allResultsForPlot["predicted_y"] = transformBack(dataScalerY, meanPredictions_transformedScale)

    PLOT_FILENAME_PREFIX =  datasetName + "_" + dataSplit + "_" + noiseType + commonSettings.getNoisePostFix(noiseType) + "_" + commonSettings.getMethodPostFix(method, OPTIMIZATION_MODE, PRE_SPECIFIED_NU) 
    
    numpy.save("all_plots/" + PLOT_FILENAME_PREFIX + "_" + str(foldId) + "_data", allResultsForPlot)
    
    return


def getSimpleLinearModel_withOneOutlierRemoved(X, y):
    outlierId_in_x = numpy.argmax(X, axis = 0)
    print("outlierId_in_x = ", outlierId_in_x)
    X = numpy.delete(X, outlierId_in_x, axis = 0)
    y = numpy.delete(y, outlierId_in_x)
    print("new X = ", X)
    print("new y = ", y)
    import sklearn
    return sklearn.linear_model.LinearRegression().fit(X, y)

def getSimpleLinearModel_withAllData(X, y):
    import sklearn
    return sklearn.linear_model.LinearRegression().fit(X, y)

def visualizePredictions(dataType, transformed_X, transformed_y, trueOutlierIndices, dataScalerX, dataScalerY, gpModel, outlierScores = None):
    assert(transformed_y.shape[0] == trueOutlierIndices.shape[0])
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt

    # transform back to original scale
    X = transformBack(dataScalerX, transformed_X)
    y = transformBack(dataScalerY, transformed_y) 

    plt.figure(figsize=(10, 5))
    lw = 2

    
    sampledX_transformedScale = getUniformSampledX_1D_forPlot(transformed_X)
    sampledX_transformedScale = commonSettings.getTorchTensor(sampledX_transformedScale)
    meanPredictions_transformedScale = gpModel.getMeanPrediction(sampledX_transformedScale)

    sampledX_originalScale = transformBack(dataScalerX, sampledX_transformedScale)

    if dataType == "syntheticFewDotData":
        print("WE ARE HERE")
        print("X.shape = ", X.shape)
        print("y.shape = ", y.shape)

        plt.figure(figsize=(10, 5))
        lw = 2
        plt.scatter(X[:,0], y, cmap="plasma")

        # print("sampledX_originalScale = ", sampledX_originalScale)

        sampledX = numpy.linspace(1.0, 5.0, 10000)
        sampledX_originalScale = numpy.reshape(sampledX, (-1, 1))

        linearModel = getSimpleLinearModel_withOneOutlierRemoved(X, y)
        # linearModel = getSimpleLinearModel_withAllData(X,y)
        plt.plot(sampledX_originalScale, linearModel.predict(sampledX_originalScale), color='red', lw=lw, label='Estimate')

        # plt.plot(transformBack(dataScalerX, sampledX_transformedScale), transformBack(dataScalerY, meanPredictions_transformedScale), color='red', lw=lw, label='Estimate')
        
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.ylim(1.0, 5.0)

        # plt.title('Example Few Dot Data')
        # plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
        # plt.savefig("all_plots/" + dataType + "_" + "noNoise" + "_" + "data_only" + ".pdf")
        # plt.savefig("all_plots/" + dataType + "_" + "noNoise" + "_" + "linearModel_trainedIncludingOutlier" + ".pdf")
        plt.savefig("all_plots/" + dataType + "_" + "noNoise" + "_" + "linearModel_trainedWithoutOutlier" + ".pdf")
        # plt.savefig("all_plots/" + dataType + "_" + noiseType + "_" + methodName + ".pdf")
        plt.show()
        assert(False)
        
    else:
        assert(dataType == "syntheticSimpleSin")
    
        if outlierScores is None:
            plt.scatter(X[trueOutlierIndices == 0,0], y[trueOutlierIndices == 0], c='b', label='inlier')
            plt.scatter(X[trueOutlierIndices == 1,0], y[trueOutlierIndices == 1], c='m', label='outlier')
        else:
            plt.scatter(X[:,0], y, c=outlierScores, cmap="plasma")
            plt.colorbar()
            

        
        
        x_plot = getScaledXData_forSyntheticSimpleSin(numpy.linspace(0, 1.0, 10000))
        trueY = getResponse_forSyntheticSimpleSin(x_plot)
        plt.plot(x_plot, trueY, color='navy', lw=lw, label='True')
        plt.plot(transformBack(dataScalerX, sampledX_transformedScale), transformBack(dataScalerY, meanPredictions_transformedScale), color='red', lw=lw, label='Estimate')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Example Nonlinear Data')
        plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
        # plt.savefig("all_plots/" + dataType + "_" + noiseType + "_" + method + ".pdf")
        plt.show()
    assert(False)

def showFewDotData():
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt

    X = numpy.asarray([[1.0], [2.0], [3.0], [4.0], [5.0], [10.0]])
    y = numpy.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 1.0])

    plt.figure(figsize=(10, 5))
    lw = 2
    plt.scatter(X[:,0], y, cmap="plasma")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Example Nonlinear Data')
    plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
    # plt.savefig("all_plots/" + dataType + "_" + noiseType + "_" + method + ".pdf")
    plt.show()
    # assert(False)

def showBowShapedData(dataType, noiseType, transformed_X, transformed_y, dataScalerX, dataScalerY):
    assert(dataType == "syntheticSimpleSin")
    assert(noiseType == "noNoise")
    import matplotlib
    matplotlib.use('tkagg')
    import matplotlib.pyplot as plt

    # transform back to original scale
    X = transformBack(dataScalerX, transformed_X)
    y = transformBack(dataScalerY, transformed_y)

    plt.figure(figsize=(10, 5))
    lw = 2
    plt.scatter(X[:,0], y, c='b')

    sampledX_transformedScale = getUniformSampledX_1D_forPlot(transformed_X)
    
    x_plot = getScaledXData_forSyntheticSimpleSin(numpy.linspace(0, 1.0, 10000))
    trueY = getResponse_forSyntheticSimpleSin(x_plot)
    plt.plot(x_plot, trueY, color='navy', lw=lw, label='True')
    # plt.plot(transformBack(dataScalerX, sampledX_transformedScale), transformBack(dataScalerY, meanPredictions_transformedScale), color='red', lw=lw, label='Estimate')
    
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title('Example Nonlinear Data')
    plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
    plt.savefig("all_plots/" + dataType + "_" + noiseType + ".pdf")
    plt.show()