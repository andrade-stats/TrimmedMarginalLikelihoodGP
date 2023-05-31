import math
import torch
import commonSettings
import gpytorch
import numpy

def getInlierAndOutliersBasedOnMarginalLikelihood_greedy(model, likelihood, X, y, maxNrOutlierSamples, greedy_method):
    assert(model.training and likelihood.training)
    assert(X.dtype is torch.float32)
    assert(y.dtype is torch.float32)

    covMatrix = likelihood(model.forward(X)).covariance_matrix  # get covariance matrix including the noise model (use forward to avoid error message "You must train on the training inputs!")
    orig_K = covMatrix.detach()
    assert(orig_K.dtype is torch.float32)

    if greedy_method == "greedy_RemoveOneByOne":
        allInlierSamplesIds, allOutlierSampleIds = greedyMaximizeMLL_RemoveOneByOne(orig_K, y, maxNrOutlierSamples)
    elif greedy_method == "greedy_RemoveBatch":
        allInlierSamplesIds, allOutlierSampleIds = greedyMaximizeMLL_RemoveBatch(orig_K, y, maxNrOutlierSamples)
    else:
        assert(False)
    
    # check that we did not miss anything
    checkIds = torch.zeros_like(y)
    checkIds[allInlierSamplesIds] = 1
    checkIds[allOutlierSampleIds] += 1
    assert(torch.equal(checkIds, torch.ones_like(y)))

    allOutlierSampleIds = allOutlierSampleIds.to(device=commonSettings.DEVICE)
    allInlierSamplesIds = allInlierSamplesIds.to(device=commonSettings.DEVICE)
    
    assert(allInlierSamplesIds.shape[0] == X.shape[0] - maxNrOutlierSamples)
    assert(allOutlierSampleIds.shape[0] == maxNrOutlierSamples)

    return allInlierSamplesIds, allOutlierSampleIds

# checked pytorch
def getMarginalLogLikelihood_fromLogDetL_and_invK(logDetK, invK, y):

    n = invK.shape[0]

    mll = n * math.log(2.0 * torch.pi)
    mll += logDetK
    mll += y @ invK @ y
    mll *= - 0.5

    return mll


def getNumpyDet(orig_K):
    
    orig_K_numpy = orig_K.detach().numpy()

    try:
        sign, logDetOrigK = numpy.linalg.slogdet(orig_K_numpy)
        if sign != 1:
            raise numpy.linalg.LinAlgError("Matrix not positive definite")
    except numpy.linalg.LinAlgError as e:
        assert(False)
        # EPSILON = 0.0001
        # e, _ = numpy.linalg.eigh(orig_K_numpy)
        # smallestEigenvalue = e[0]
        # correctionEpsilon = numpy.abs(smallestEigenvalue) + EPSILON
        # sign, logDetOrigK = numpy.linalg.slogdet(orig_K_numpy + correctionEpsilon * numpy.eye_like(orig_K_numpy))

    assert(sign == 1)

    return logDetOrigK



# checked pytorch
# returns allLOO_mll (an array of size n), with
# allLOO_mll[i] = marginal log likelihood of model when sample i is removed
def calculateAllMLL_oneRemoved(orig_K, orig_y):

    commonSettings.assertOnDevice(orig_K)
    commonSettings.assertOnDevice(orig_y)

    n = orig_K.shape[0]
    assert(n == orig_y.shape[0])

    invOrigK = torch.linalg.inv(orig_K)

    try:
        eig_values, _ = gpytorch.diagonalization(orig_K)
        logDetOrigK = torch.sum(torch.log(eig_values))
    except torch._C._LinAlgError:
        logDetOrigK = getNumpyDet(orig_K)
        logDetOrigK = commonSettings.getTorchTensor(logDetOrigK)

    commonSettings.assertOnDevice(invOrigK)
    commonSettings.assertOnDevice(logDetOrigK)
    
    lastId = n - 1

    allLOO_mll = torch.zeros(n, device=commonSettings.DEVICE)

    commonSettings.assertOnDevice(allLOO_mll)

    for i in range(n):

        K = torch.clone(orig_K.detach())
        invK = torch.clone(invOrigK.detach())
        y = torch.clone(orig_y.detach())

        commonSettings.assertOnDevice(K)
        commonSettings.assertOnDevice(invK)
        commonSettings.assertOnDevice(y)

        # ****** permutation of rows and columns *************************

        K[[i, lastId], : ] = K[[lastId, i], : ]
        K[: , [i, lastId]] = K[:, [lastId, i]]

        invK[[i, lastId], : ] = invK[[lastId, i], : ]
        invK[: , [i, lastId]] = invK[:, [lastId, i]]

        y[[lastId, i]] = y[[i, lastId]]

        # ******** fast inverse *******************************************

        U = invK[0:lastId, 0:lastId]
        v = invK[lastId,0:lastId]
        b = K[lastId,0:lastId]

        fac = 1.0 / (1.0 - (b @ v))
        invA = U + fac * torch.outer((U @ b), v)

        # ******* fast determinant ********************************************

        logDetA = logDetOrigK + torch.log(invK[lastId,lastId])

        # ****** calculate MLL **************
        allLOO_mll[i] = getMarginalLogLikelihood_fromLogDetL_and_invK(logDetA, invA, y[0:lastId])

    return allLOO_mll


def greedyMaximizeMLL_RemoveOneByOne(orig_K, orig_y, maxNrOutlierSamples):

    allInlierIds = torch.arange(orig_K.shape[0])
    allOutlierIds = torch.zeros(maxNrOutlierSamples, dtype = torch.long, device=commonSettings.DEVICE)

    for i in range(maxNrOutlierSamples):
        allLOO_mll = calculateAllMLL_oneRemoved(orig_K, orig_y)

        outlierId = torch.argmax(allLOO_mll) # find sample which leads to highest MLL when removed
        
        allOutlierIds[i] = allInlierIds[outlierId]

        assert(allInlierIds.shape[0] == orig_y.shape[0] and allInlierIds.shape[0] == orig_K.shape[0] and allInlierIds.shape[0] == orig_K.shape[1])
        all_current_local_ids = torch.arange(allInlierIds.shape[0])

        allInlierIds = allInlierIds[all_current_local_ids != outlierId]
        orig_y = orig_y[all_current_local_ids != outlierId]
        orig_K = orig_K[all_current_local_ids != outlierId, :]
        orig_K = orig_K[:, all_current_local_ids != outlierId]

        assert(orig_K.shape[0] == allInlierIds.shape[0] and orig_K.shape[1] == allInlierIds.shape[0])

    return allInlierIds, allOutlierIds

def greedyMaximizeMLL_RemoveBatch(orig_K, orig_y, maxNrOutlierSamples):

    n = orig_K.shape[0]

    allLOO_mll = calculateAllMLL_oneRemoved(orig_K, orig_y)
    assert(allLOO_mll.shape[0] == n)

    _, sorted_ids = torch.sort(allLOO_mll)
    allInlierIds = sorted_ids[0:n - maxNrOutlierSamples]
    allOutlierIds = sorted_ids[n - maxNrOutlierSamples:n]

    assert(allOutlierIds.shape[0] == maxNrOutlierSamples)
    return allInlierIds, allOutlierIds
