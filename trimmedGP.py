

import torch
import gpytorch
import time
import fastGreedy_pytorch
import commonSettings
from commonSettings import EstimationType

import commons_GP
import projectedGradientDescent
import sklearn.model_selection
    
def residualNuTrimmedGP(full_X, full_y, maxNrOutlierSamples, method):
    assert(full_X.shape[0] == full_y.shape[0])
    FULL_N = full_X.shape[0]
    NU = maxNrOutlierSamples / FULL_N 
    MIN_NR_INLIERS = FULL_N - maxNrOutlierSamples

    print("NU = ", NU)

    NR_FOLDS = 10

    NU_FOR_CV = NU / (1.0 - (1.0 / NR_FOLDS))
    print("NU_FOR_CV = ", NU_FOR_CV)
    
    cv = sklearn.model_selection.KFold(n_splits=NR_FOLDS, random_state=4323, shuffle=True)
    residuals_abs = torch.zeros(FULL_N)

    for i, (train_index, valid_index) in enumerate(cv.split(full_X)):
        cv_maxNrOutliers = int(train_index.shape[0] * NU_FOR_CV)
        if method == "projectedGradient" or method.startswith("greedy"):
            _, _, gpModel, _, _ = trainTrimmedGP(full_X[train_index, :], full_y[train_index], cv_maxNrOutliers, commonSettings.getSigmaEstimationTypes(), method)
        elif method == "LMDbaseline":
            assert(False)
            # gpModel = LMD(full_X[train_index, :], full_y[train_index], cv_maxNrOutliers)
        else:
            assert(False)

        mean_predictions = gpModel.getMeanPrediction(full_X[valid_index, :])
        assert(valid_index.shape[0] == mean_predictions.shape[0])
        residuals_abs[valid_index] = torch.abs(mean_predictions - full_y[valid_index])
    
    inlierAbsDiff, _ = torch.sort(residuals_abs)[0:MIN_NR_INLIERS]
    sigmaEstimate = commons_GP.getAsymptoticCorrectedSigma(inlierAbsDiff, FULL_N)
    
    maxNrOutlierSamples_new = torch.sum(residuals_abs > sigmaEstimate * 2)

    allPValues_logScale = {}
    allEstimatedNoiseVariances = {}
    standardizedResiduals = {}
    sigmaEstimateType = EstimationType.LIKELIHOOD_NOISE

    pValues_logScale = commons_GP.getLogPValues_fromSigmaEstimate_and_absValues(residuals_abs, sigmaEstimate)
    allPValues_logScale[sigmaEstimateType] = pValues_logScale
    allEstimatedNoiseVariances[sigmaEstimateType] = sigmaEstimate ** 2

    standardizedResiduals[sigmaEstimateType] = (residuals_abs / sigmaEstimate).cpu().numpy()

    return maxNrOutlierSamples_new, allPValues_logScale, allEstimatedNoiseVariances, standardizedResiduals


def getFinalModelForPrediction(full_X, full_y, maxNrOutlierSamples, all_sigmaEstimates, method):

    if method == "projectedGradient" or method.startswith("greedy"):
         _, _, gpModel, _, _ = trainTrimmedGP(full_X, full_y, maxNrOutlierSamples, commonSettings.getSigmaEstimationTypes(), method)
    elif method == "LMDbaseline":
        assert(False)
        # gpModel = LMD(full_X, full_y, maxNrOutlierSamples)
    else:
        assert(False)

    sigmaEstimate = all_sigmaEstimates[EstimationType.LIKELIHOOD_NOISE]

    if sigmaEstimate > commons_GP.LOWER_BOUND_ON_SIGMA:
        gpModel.likelihood.noise = sigmaEstimate ** 2
    else:
        gpModel.likelihood.noise = commons_GP.LOWER_BOUND_ON_SIGMA ** 2

    return gpModel


def set_new_data_and_get_loss(model, mll, X, y, allInlierSamplesIds):
    
    dataForTrainingHyperparameters_X = X[allInlierSamplesIds, :]
    dataForTrainingHyperparameters_y = y[allInlierSamplesIds]
    model.set_train_data(inputs=dataForTrainingHyperparameters_X, targets=dataForTrainingHyperparameters_y, strict = False)

    # Calc loss and backprop gradients
    loss = -mll(model(dataForTrainingHyperparameters_X), dataForTrainingHyperparameters_y) # note: mll returns marginal log-likelihood divided by the number of samples

    return loss.item()


# checked
def trainTrimmedGP(X, y, maxNrOutlierSamples, sigmaEstimateTypes, optimizationMode):
    assert(X.shape[0] == y.shape[0])
    FULL_N = X.shape[0]
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = commons_GP.ExactGPModel(X, y, likelihood)
    likelihood.noise = commons_GP.INITIAL_SIGMA_SQUARE

    model = commonSettings.setDataType(model)
    likelihood = commonSettings.setDataType(likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    

    # Use the adam optimizer
    def getOptimizer(model):
        return torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    optimizer = getOptimizer(model)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # ***** proposed Algorthm for joint optimization, corresponds to Algorithm 1 in paper *****

    previous_inlier_ids = None
    previous_loss = torch.inf 
    
    i = 0
    while(True):
        startTime = time.time()

        # **** !!! must set model and likelihood to train() otherwise covariance matrix for predictive inference is calculated !!! ****
        assert(model.training and likelihood.training)
        
        if optimizationMode == "projectedGradient":
            # proposed method in UAI 2023 paper
            allInlierSamplesIds, allOutlierSampleIds = projectedGradientDescent.getInlierAndOutliersBasedOnMarginalLikelihood_hardthresholding(model, likelihood, X, y, maxNrOutlierSamples)
        elif optimizationMode.startswith("greedy"):
            # not so naive implementation of a greedy method, but much slower than the projected gradient descent method
            allInlierSamplesIds, allOutlierSampleIds  = fastGreedy_pytorch.getInlierAndOutliersBasedOnMarginalLikelihood_greedy(model, likelihood, X, y, maxNrOutlierSamples, optimizationMode)
        else:
            assert(False)
        
        allInlierSamplesIds = torch.sort(allInlierSamplesIds)[0]
        assert(allOutlierSampleIds.shape[0] == maxNrOutlierSamples)
        
        if (previous_inlier_ids is None) or not torch.equal(allInlierSamplesIds, previous_inlier_ids):
            print("** inlier set changed ** iteration nr = ", i)
            
            new_loss = set_new_data_and_get_loss(model, mll, X, y, allInlierSamplesIds)
            if new_loss >= previous_loss:
                # new set does not improve marginal likelihood -> fallback on previous inliers
                assert(previous_inlier_ids is not None)
                allInlierSamplesIds = previous_inlier_ids
            else:
                previous_inlier_ids = allInlierSamplesIds
                optimizer = getOptimizer(model)
            
            model.set_train_data(inputs=X[allInlierSamplesIds, :], targets=y[allInlierSamplesIds], strict = False)
                
        
        optimizer.zero_grad() # Zero gradients from previous iteration

        # Calc loss and backprop gradients
        previous_loss = -mll(model(X[allInlierSamplesIds, :]), y[allInlierSamplesIds]) # note: mll returns marginal log-likelihood divided by the number of samples

        previous_loss.backward()
        commons_GP.showProgressGP(i, previous_loss, model, likelihood,  (time.time() - startTime))
        optimizer.step()
        
        new_loss = -mll(model(X[allInlierSamplesIds, :]), y[allInlierSamplesIds]).item()

        i = i + 1

        if new_loss >= previous_loss:
            break

    
    print("**************************************")
    print("total number of iterations = ", i)
    print("**************************************")
   

    # ***** simple variance estimation (no CV) and scoring of potential outliers *****

    allPValues_logScale = {}
    allEstimatedNoiseVariances = {}
    

    gpModel = commons_GP.BasicGP(model, likelihood)

    assert(len(sigmaEstimateTypes) == 1)
    for sigmaEstimateType in sigmaEstimateTypes:
        
        sigmaEstimate, current_average_MLL = commons_GP.setSigma(gpModel)
        
        possibleOutliers_X = X[allOutlierSampleIds, :]
        possibleOutliers_y = y[allOutlierSampleIds]
        predictions_at_trainingDataPoints = commons_GP.getPredictions(model, likelihood, possibleOutliers_X)
        meanPredictions = commons_GP.getMeanPredictions(predictions_at_trainingDataPoints)
        centeredAbsValues = torch.abs(meanPredictions - possibleOutliers_y.detach())
        pValues_onlyOutliers_logScale = commons_GP.getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate)
        
        print("sigmaEstimateType = ", sigmaEstimateType)
        print("pValues_onlyOutliers = ", pValues_onlyOutliers_logScale)
        assert(torch.all(pValues_onlyOutliers_logScale < 0.0))
        pValues_logScale = torch.zeros_like(y)
        pValues_logScale[allOutlierSampleIds] = pValues_onlyOutliers_logScale

        allPValues_logScale[sigmaEstimateType] = pValues_logScale
        allEstimatedNoiseVariances[sigmaEstimateType] = sigmaEstimate ** 2

    allOutlierSampleIds_detected_zeroOne = torch.zeros(y.shape[0])
    allOutlierSampleIds_detected_zeroOne[allOutlierSampleIds] = 1

    return allPValues_logScale, allEstimatedNoiseVariances, gpModel, current_average_MLL, allOutlierSampleIds_detected_zeroOne


