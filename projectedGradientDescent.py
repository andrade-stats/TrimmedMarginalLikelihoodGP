
import numpy
import gpytorch
import torch
import commonSettings
import scipy.sparse.linalg

def project_b(b, maxNrOutlierSamples):
    n = b.shape[0]
    zero_ids = torch.argsort(torch.abs(b))[0:(n - maxNrOutlierSamples)]
    b[zero_ids] = 0.0
    return zero_ids

def evalObjectiveFunction(invOrigK, y, b):
    return (y + b) @ (invOrigK @ (y + b))
    

def getInlierAndOutliersBasedOnMarginalLikelihood_hardthresholding(model, likelihood, X, y, maxNrOutlierSamples):
    assert(model.training and likelihood.training)
    assert(X.dtype is torch.float32)
    assert(y.dtype is torch.float32)

    covMatrix = likelihood(model.forward(X)).covariance_matrix  # get covariance matrix including the noise model (use forward to avoid error message "You must train on the training inputs!")
    orig_K = covMatrix.detach()
    assert(orig_K.dtype is torch.float32)

    print("*** start running inversion (HERE !) ****")

    n = orig_K.shape[0]
    
    invOrigK = gpytorch.root_inv_decomposition(orig_K)

    try:
        eig_values, _ = gpytorch.diagonalization(orig_K)
    except torch._C._LinAlgError:
        try:
            eig_values, _ = scipy.sparse.linalg.eigsh(orig_K.detach().numpy(), k=1, which = "SM")
        except scipy.sparse.linalg._eigen.arpack.ArpackNoConvergence:
            eig_values, _  = numpy.linalg.eigh(orig_K.detach().numpy())
            
    smallest_eigenvalue_origK = eig_values[0]
    
    lipschitzConstant = 2.0 * (1.0 / smallest_eigenvalue_origK)
    invLipschitzConstant = (1.0 / lipschitzConstant)
    print("lipschitzConstant = ", lipschitzConstant)
    print("invLipschitzConstant = ", invLipschitzConstant)
    print("maxNrOutlierSamples = ", maxNrOutlierSamples)

    b = - y 

    print("f(b) = ", evalObjectiveFunction(invOrigK, y, b))

    previous_b = torch.zeros(n, device=commonSettings.DEVICE)
    
    converged = False

    for i in range(200):
        grad_b = 2.0 * (invOrigK @ (y + b))

        b = b - invLipschitzConstant * grad_b
        allInlierSamplesIds = project_b(b, maxNrOutlierSamples)
        
        diff_to_previous = torch.sum(torch.square(previous_b - b))
        # print("f(b) = ", evalObjectiveFunction(invOrigK, y, b))
        previous_b = torch.clone(b)

        if diff_to_previous < 0.00000001:
            converged = True
            break
        

    allOutlierSampleIds = numpy.arange(n)
    allOutlierSampleIds = numpy.delete(allOutlierSampleIds, allInlierSamplesIds.cpu().numpy())
    
    allOutlierSampleIds = torch.from_numpy(allOutlierSampleIds)
    allOutlierSampleIds = allOutlierSampleIds.to(device=commonSettings.DEVICE)
    allInlierSamplesIds = allInlierSamplesIds.to(device=commonSettings.DEVICE)
    
    assert(allInlierSamplesIds.shape[0] == n - maxNrOutlierSamples)
    assert(allOutlierSampleIds.shape[0] == maxNrOutlierSamples)
    
    return allInlierSamplesIds, allOutlierSampleIds
