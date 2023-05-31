import math
import torch
import gpytorch
import numpy

import metrics_GP
import scipy.stats

import commonSettings
from commonSettings import EstimationType

LOWER_BOUND_ON_SIGMA = math.sqrt(1.000E-04)   # set by gpytorch

# important hyper-parameters
INITIAL_SIGMA_SQUARE = 10.0
MIN_STANDARD_NR_TRAINING_ITERATIONS = 1000
MAX_STANDARD_NR_TRAINING_ITERATIONS = 20000

MEAN_FUNCTION = gpytorch.means.ZeroMean()

def COVARIANCE_FUNCTION(train_x):
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = train_x.shape[1]))
    
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MEAN_FUNCTION
        self.covar_module = COVARIANCE_FUNCTION(train_x)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, meanFunc = MEAN_FUNCTION, covFunc = COVARIANCE_FUNCTION):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(train_x.shape[0])
        # use all training points as inducing points
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(VariationalGPModel, self).__init__(variational_strategy)

        self.mean_module = meanFunc
        self.covar_module = covFunc(train_x)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred



class BasicGP:
    def __init__(self, model, likelihood):
        self.model = model
        self.likelihood = likelihood
        
    def getMeanPrediction(self, X):

        if len(X.shape) == 1:
            X = X.view(1, -1)
        
        predictive_distribution_at_X = getPredictions(self.model, self.likelihood, X)
        return getMeanPredictions(predictive_distribution_at_X)
    

    def aggregateSamples(eval_result):
        if len(eval_result.shape) == 2:
            print("RUN MEAN")
            return torch.mean(eval_result, dim = 0)
        else:
            return eval_result

    

    def evaluatePredictions(self, X, true_y):
        
        if len(X.shape) == 1:
            X = X.view(1, -1)
        
        try:
            
            predictive_distribution_at_X = getPredictions(self.model, self.likelihood, X)
            
            nll = metrics_GP.negative_log_predictive_density(predictive_distribution_at_X, true_y) # average negative log likelihood
            msll = metrics_GP.mean_standardized_log_loss(predictive_distribution_at_X, true_y) # as in "Gaussian Processes for Machine Learning", page 23 (41 pdf)

            meanPredictions = getMeanPredictions(predictive_distribution_at_X)
            rmse = metrics_GP.root_mean_squared_error(meanPredictions, true_y)
            median_absolute_error = metrics_GP.median_absolute_error(meanPredictions, true_y)
        
        except ValueError:
            # sometimes necessary due to numerical instability of GP student-t 
            
            all_nll = torch.zeros(X.shape[0])
            all_msll = torch.zeros(X.shape[0])
            all_rmse = torch.zeros(X.shape[0])
            all_median_absolute_error = torch.zeros(X.shape[0])

            for i in range(X.shape[0]):
                X_one = (X[i, :]).reshape(1, -1)
                y_true_one = true_y[i].reshape(-1)
                
                predictive_distribution_at_X = getPredictions(self.model, self.likelihood, X_one)
            
                all_nll[i] = metrics_GP.negative_log_predictive_density(predictive_distribution_at_X, y_true_one) # average negative log likelihood
                all_msll[i] = metrics_GP.mean_standardized_log_loss(predictive_distribution_at_X, y_true_one) # as in "Gaussian Processes for Machine Learning", page 23 (41 pdf)

                meanPredictions = getMeanPredictions(predictive_distribution_at_X)
                all_rmse[i] = metrics_GP.root_mean_squared_error(meanPredictions, y_true_one)
                all_median_absolute_error[i] = metrics_GP.median_absolute_error(meanPredictions, y_true_one)

            nll = torch.mean(all_nll)
            msll = torch.mean(all_msll)
            rmse = torch.mean(all_rmse)
            median_absolute_error = torch.median(all_median_absolute_error)

        return nll, msll, rmse, median_absolute_error

    def getMLL(self, X_new = None, y_new = None):
        if X_new is not None:
            self.model.set_train_data(inputs=X_new, targets=y_new, strict = False)

        self.model.train()
        self.likelihood.train()

        assert(self.model.training and self.likelihood.training)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        mll = mll.to(device=commonSettings.DEVICE)

        X = self.model.train_inputs[0]
        y = self.model.train_targets
        return mll(self.model(X), y).item()





# checked
def getPredictions(model, likelihood, X):

    model.eval()
    likelihood.eval()
    
    # number of samples used for estimating the integral of the liklihood =  int_f p(y | f) p(f) df,
    # where p(f) is a multivariate gaussian, and p(y | f) is the likelihood (e.g. student t)
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(100):
        predictive_distribution = likelihood(model(X))
        
    return predictive_distribution

# get E[y | x]
def getMeanPredictions(predictive_distribution):
    meanPredictions = predictive_distribution.loc.detach()
    if len(meanPredictions.shape) == 2:
        meanPredictions = torch.mean(meanPredictions, axis = 0)
    return meanPredictions


def setSigma(gpModel):

    sigmaEstimate = torch.sqrt(gpModel.likelihood.noise).item()

    current_average_MLL = gpModel.getMLL()

    return sigmaEstimate, current_average_MLL




# checked
def showProgressGP(i, loss, model, likelihood, runtime = 0.0):
    # numpy.set_printoptions(precision=2)
    lengthscaleOutput = model.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0,:]

    # note that likelihood.noise.item() correponds to sigma^2

    if (i <= 100) or (i % 100 == 0):
        if hasattr(likelihood, 'deg_free'):
            # student-t likelihood
            print(f"Iter {i} - Loss: {loss.item():.3f} outputscale: {model.covar_module.outputscale.item():.3f}    lengthscale: {lengthscaleOutput}   noise variance: {likelihood.noise.item():.3f}   nu: {likelihood.deg_free.item():.3f}")
        
        else:
            # Normal likelihood
            print(f"Iter {i} - Loss: {loss.item():.3f} outputscale: {model.covar_module.outputscale.item():.3f}    lengthscale: {lengthscaleOutput}   noise variance: {likelihood.noise.item():.3f}  (runtime = {(runtime / 60.0):.3f})")
        
    return

# checked
def standardTraining(model, likelihood, mll, X, y, min_training_iter = MIN_STANDARD_NR_TRAINING_ITERATIONS, learningRate = 0.1):

    # Use the adam optimizer
    if type(model) is ExactGPModel:
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)  # Includes GaussianLikelihood parameters
    else:
        assert(type(model) is VariationalGPModel)
        optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=learningRate) # need to include likelihood explicitly
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    previous_loss = float("inf")
    
    for i in range(MAX_STANDARD_NR_TRAINING_ITERATIONS):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Calc loss and backprop gradients
        loss = -mll(model(X), y) # note: mll returns marginal log-likelihood divided by the number of samples
       
        loss.backward()
        showProgressGP(i, loss, model, likelihood)
        optimizer.step()

        if loss >= previous_loss and i >= min_training_iter:
            break
        else:
            previous_loss = loss
    
    assert(i < MAX_STANDARD_NR_TRAINING_ITERATIONS - 1) # if this fails, then this suggests an issue with convergence

    average_mll_value = mll(model(X), y).item()
    marginalLikelihood_value = average_mll_value  * y.shape[0]
    return model, likelihood, marginalLikelihood_value, average_mll_value




def classifyBasedOnScores(outlierScores, maxNrOutlierSamples):
    _, outlierIds = torch.sort(- outlierScores)[0:maxNrOutlierSamples]
    outliers_zeroOne = torch.zeros_like(outlierScores)
    outliers_zeroOne[outlierIds] = 1
    return outliers_zeroOne




# checked
# corresponds to Standard-GP in paper
def trainVanillaGP(X, y, sigmaEstimateTypes):

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X, y, likelihood)
    likelihood.noise = INITIAL_SIGMA_SQUARE

    model = commonSettings.setDataType(model)
    likelihood = commonSettings.setDataType(likelihood)
    

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model, likelihood, _, average_mll_value = standardTraining(model, likelihood, mll, X, y)
    learnedGP = BasicGP(model, likelihood)

    allPValues_logScale = {}
    allEstimatedNoiseVariances = {}
    standardizedResiduals = {}

    for sigmaEstimateType in sigmaEstimateTypes:
        
        # note that likelihood.noise.item() correponds to sigma^2
        estimatedNoiseVariance = likelihood.noise.item()

        assert(sigmaEstimateType == EstimationType.LIKELIHOOD_NOISE)
        meanPredictions = learnedGP.getMeanPrediction(X)
        centeredAbsValues = torch.abs(meanPredictions - y.detach())

        sigmaEstimate = math.sqrt(estimatedNoiseVariance)
        allPValues_logScale[sigmaEstimateType] = getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate)
        allEstimatedNoiseVariances[sigmaEstimateType] = estimatedNoiseVariance
        standardizedResiduals[sigmaEstimateType] = (centeredAbsValues / sigmaEstimate).cpu().numpy()

    return allPValues_logScale, allEstimatedNoiseVariances, learnedGP, average_mll_value, standardizedResiduals


# simple baseline that filters out large squared y-values
def LMD(X, y, maxNrOutlierSamples):
    nr_inliers = X.shape[0] - maxNrOutlierSamples
    allInlierSamplesIds = torch.argsort(torch.square(y))[0:nr_inliers]
    allOutlierSampleIds = torch.argsort(torch.square(y))[nr_inliers:y.shape[0]]
    assert(allInlierSamplesIds.shape[0] + allOutlierSampleIds.shape[0] == y.shape[0])
    _, _, gpModel, _, _= trainVanillaGP(X[allInlierSamplesIds, :], y[allInlierSamplesIds], sigmaEstimateTypes = commonSettings.getSigmaEstimationTypes())


    allPValues_logScale = {}
    allEstimatedNoiseVariances = {}
    standardizedResiduals = {}

    for sigmaEstimateType in commonSettings.getSigmaEstimationTypes():
        
        # note that likelihood.noise.item() correponds to sigma^2
        estimatedNoiseVariance = gpModel.likelihood.noise.item()

        assert(sigmaEstimateType == EstimationType.LIKELIHOOD_NOISE)
        meanPredictions = gpModel.getMeanPrediction(X)
        centeredAbsValues = torch.abs(meanPredictions - y.detach())

        sigmaEstimate = math.sqrt(estimatedNoiseVariance)
        allPValues_logScale[sigmaEstimateType] = getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate)
        allEstimatedNoiseVariances[sigmaEstimateType] = estimatedNoiseVariance
        standardizedResiduals[sigmaEstimateType] = (centeredAbsValues / sigmaEstimate).cpu().numpy()

    return allInlierSamplesIds, allOutlierSampleIds, allPValues_logScale, allEstimatedNoiseVariances, gpModel, standardizedResiduals


def getBestGamma(X, y):
    # ALL_GAMMA_CANDIDATES = [0.01, 0.05, 0.1, 0.3, 0.5]
    bestGamma = 0.1 # preliminary experiments suggested that this is a good value
    return bestGamma

# checked
# here gamma is the one from "Variational Inference based on Robust Divergences"
# reasonable values are between 0.1 and 0.9
def trainGP_withGammaDivergence(X, y, sigmaEstimateTypes):

    bestGamma = getBestGamma(X, y)
    assert(bestGamma > 0.0 and bestGamma < 1.0)
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = VariationalGPModel(X)
    likelihood.noise = INITIAL_SIGMA_SQUARE

    model = commonSettings.setDataType(model)
    likelihood = commonSettings.setDataType(likelihood)

    # note that gamma(GammaRobustVariationalELBO) = 1 + gamma("Variational Inference based on Robust Divergences") 
    mll = gpytorch.mlls.GammaRobustVariationalELBO(likelihood, model, num_data=y.shape[0], gamma=bestGamma + 1.0)

    model, likelihood, _, average_mll_value = standardTraining(model, likelihood, mll, X, y, min_training_iter = MIN_STANDARD_NR_TRAINING_ITERATIONS, learningRate = 0.1)
    learnedGP = BasicGP(model, likelihood)

    allPValues_logScale = {}
    allEstimatedNoiseVariances = {}
    standardizedResiduals = {}

    for sigmaEstimateType in sigmaEstimateTypes:
        
         # note that likelihood.noise.item() correponds to sigma^2
        estimatedNoiseVariance = likelihood.noise.item()

        assert(sigmaEstimateType == EstimationType.LIKELIHOOD_NOISE)
        meanPredictions = learnedGP.getMeanPrediction(X)
        centeredAbsValues = torch.abs(meanPredictions - y.detach())        
        sigmaEstimate = math.sqrt(estimatedNoiseVariance)

        allPValues_logScale[sigmaEstimateType] = getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate)
        allEstimatedNoiseVariances[sigmaEstimateType] = estimatedNoiseVariance
        standardizedResiduals[sigmaEstimateType] = (centeredAbsValues / sigmaEstimate).cpu().numpy()

    return allPValues_logScale, allEstimatedNoiseVariances, learnedGP, average_mll_value, standardizedResiduals


# checked
def trainStudentTGP(X, y, sigmaEstimateTypes):

    # initialize likelihood and model
    model = VariationalGPModel(X)
    likelihood = gpytorch.likelihoods.StudentTLikelihood()
    likelihood.noise = INITIAL_SIGMA_SQUARE

    model = commonSettings.setDataType(model)
    likelihood = commonSettings.setDataType(likelihood)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.shape[0])

    model, likelihood, _, average_mll_value = standardTraining(model, likelihood, mll, X, y)
    learnedGP = BasicGP(model, likelihood)

    allPValues_logScale = {}
    allEstimatedNoiseVariances = {}
    standardizedResiduals = {}

    for sigmaEstimateType in sigmaEstimateTypes:

        assert(sigmaEstimateType == EstimationType.LIKELIHOOD_NOISE)
        allPValues_logScale[sigmaEstimateType], allEstimatedNoiseVariances[sigmaEstimateType], standardizedResiduals[sigmaEstimateType] = getRobustEstimates(learnedGP, X, y)

    return allPValues_logScale, allEstimatedNoiseVariances, learnedGP, average_mll_value, standardizedResiduals


def getRobustEstimates(learnedGP, X, y):
    meanPredictions = learnedGP.getMeanPrediction(X)
    centeredAbsValues = torch.abs(meanPredictions - y.detach())
    
    correctionFactor = math.sqrt(1.0 / scipy.stats.chi2.ppf(0.5, df = 1.0))
    correctedSigmaEstimate = correctionFactor * torch.median(centeredAbsValues)
    correctedSigmaEstimate = correctedSigmaEstimate.item()
    logPValues = getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate = correctedSigmaEstimate)
    noiseVariance = correctedSigmaEstimate ** 2
    standardizedResiduals = centeredAbsValues / correctedSigmaEstimate

    return logPValues, noiseVariance, standardizedResiduals.cpu().numpy()


def getAllRobustEstimates(learnedGP, X, y):
    standardizedResiduals = {}

    for sigmaEstimateType in commonSettings.getSigmaEstimationTypes():
        assert(sigmaEstimateType == EstimationType.LIKELIHOOD_NOISE)
        _, _, standardizedResiduals[sigmaEstimateType] = getRobustEstimates(learnedGP, X, y)

    return standardizedResiduals


# checked
def getPValues_fromStudentT_model(predictions_at_trainingDataPoints, observed_y):
    centeredAbsValues = numpy.abs((predictions_at_trainingDataPoints.loc - observed_y).numpy())
    studentT = scipy.stats.t(loc = numpy.zeros_like(centeredAbsValues), df = predictions_at_trainingDataPoints.df.numpy(), scale = predictions_at_trainingDataPoints.scale.numpy())
    
    pValuesEachSample = 2.0 * studentT.cdf(- centeredAbsValues)
    pValues = numpy.mean(pValuesEachSample, axis = 0)

    return pValues



# checked
def getPValues_fromGP_model(predictions_at_trainingDataPoints, observed_y):

    allScales = (torch.sqrt(torch.diag(predictions_at_trainingDataPoints.covariance_matrix))).detach().numpy()

    centeredAbsValues = numpy.abs((predictions_at_trainingDataPoints.loc - observed_y).detach().numpy())
    normal = scipy.stats.norm(loc = numpy.zeros_like(centeredAbsValues), scale = allScales)
    
    pValues = 2.0 * normal.cdf(- centeredAbsValues)

    return pValues




def getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate):
    normal = scipy.stats.norm(loc = numpy.zeros_like(centeredAbsValues.cpu().numpy()), scale = sigmaEstimate)
    log_pValues = math.log(2.0) + normal.logcdf(- centeredAbsValues.cpu().numpy())
    log_pValues = torch.from_numpy(log_pValues)
    log_pValues = log_pValues.float()
    log_pValues = log_pValues.to(device=commonSettings.DEVICE)
    return log_pValues

# estimates the scale (sigma) of the random noise, which is assumed to be gaussian
def getCorrectedSigmaEstimate(likelihood, predictions_at_trainingDataPoints, observed_y, maxNrOutlierSamples = None):

    observed_y = observed_y.detach()

    # get E[y | x]
    meanPredictions = getMeanPredictions(predictions_at_trainingDataPoints)
    

    if type(likelihood) is gpytorch.likelihoods.GaussianLikelihood:
        if maxNrOutlierSamples is None:
            # no correction needed
            estimatedNoiseVariance = likelihood.noise.item()
            correctedSigmaEstimate = numpy.sqrt(estimatedNoiseVariance)
            
            estimatedVarFromPredictions = numpy.mean(numpy.square(meanPredictions - observed_y))
            print("estimatedVarFromPredictions = ", estimatedVarFromPredictions)
            print("estimatedNoiseVariance = ", estimatedNoiseVariance)
        else:
            # use asymptotic correction
            centeredAbsValues = numpy.abs(meanPredictions - observed_y)
            n = observed_y.shape[0]
            inlierAbsDiff = numpy.sort(centeredAbsValues)[0:(n - maxNrOutlierSamples)]
            correctedSigmaEstimate = getAsymptoticCorrectedSigma(inlierAbsDiff, n) 
    else:
        assert(type(likelihood) is gpytorch.likelihoods.StudentTLikelihood)
        # use MAD (median absolute deviation) with asymptotic correction
        centeredAbsValues = numpy.abs(meanPredictions - observed_y)
        correctionFactor = numpy.sqrt(1.0 / scipy.stats.chi2.ppf(0.5, df = 1.0))
        correctedSigmaEstimate = correctionFactor * numpy.median(centeredAbsValues)
        
    return correctedSigmaEstimate


# checked (from sigmaCorrectionMethods.py)
def getAsymptoticCorrectedSigma(inlierAbsDiff, n):

    m = inlierAbsDiff.shape[0]
    inlierRatio = m / n

    if m == n:
        return torch.sqrt(torch.mean(torch.square(inlierAbsDiff))).item()
    else:
        correctionFactor = 1.0 / scipy.stats.chi2.ppf(inlierRatio, df = 1.0)

        empiricalQuantile = torch.max(inlierAbsDiff)
        
        correctedSigma = empiricalQuantile * torch.sqrt(correctionFactor)
        return correctedSigma.item()