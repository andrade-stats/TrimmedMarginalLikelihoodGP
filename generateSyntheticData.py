
import numpy
import scipy.stats
import commons_data_preparation

def generateDataSimple(NR_SAMPLES, OUTLIER_RATIO):
    assert(OUTLIER_RATIO >= 0.0 and OUTLIER_RATIO <= 0.25)
    numpy.random.seed(3523421)
    
    NR_OUTLIER_SAMPLES = int(NR_SAMPLES * OUTLIER_RATIO)
    NR_INLIER_SAMPLES = NR_SAMPLES - NR_OUTLIER_SAMPLES
    
    INLIER_MEAN = 2.0
    INLIER_STD = 0.5
    
    OUTLIER_MEAN = -2.0
    OUTLIER_STD = 0.1
    
    trueInlierSamples = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_INLIER_SAMPLES)
    trueOutlierSamples = scipy.stats.norm.rvs(loc=OUTLIER_MEAN, scale=OUTLIER_STD, size=NR_OUTLIER_SAMPLES)
    
    allSamples = numpy.hstack((trueOutlierSamples, trueInlierSamples))
    trueOutlierIndices = numpy.zeros(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES, dtype = numpy.int)
    trueOutlierIndices[0:NR_OUTLIER_SAMPLES] = 1
    
    # shuffle samples
    rndPos = numpy.arange(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES)
    numpy.random.shuffle(rndPos)
    allSamples = allSamples[rndPos]
    trueOutlierIndices = trueOutlierIndices[rndPos]
    
    return INLIER_MEAN, INLIER_STD, allSamples.astype(numpy.float32), trueOutlierIndices, trueInlierSamples.astype(numpy.float32), trueOutlierSamples.astype(numpy.float32)

def getTruncatedNormSamples(lowerBound, upperBound, mean, std, totalSize):

    allSamples = []
    while(len(allSamples) < totalSize):
        sample = scipy.stats.norm.rvs(loc=mean, scale=std, size=1)[0]
        if sample > lowerBound and sample < upperBound:
            allSamples.append(sample)
    
    return numpy.asarray(allSamples)

    
def generateDataAdvanced(NR_SAMPLES, OUTLIER_RATIO, useTruncated = True):
    assert(OUTLIER_RATIO >= 0.0 and OUTLIER_RATIO <= 0.25)
    numpy.random.seed(3523421)
    
    NR_OUTLIER_SAMPLES = int(NR_SAMPLES * OUTLIER_RATIO)
    NR_INLIER_SAMPLES = NR_SAMPLES - NR_OUTLIER_SAMPLES
    
    # DISTANCE_FROM_ZERO = 1000.5
    # DISTANCE_FROM_ZERO = 1.5
    DISTANCE_FROM_ZERO = 1.5
    
    OFFSET = 50.5
    # OFFSET = 0.0
    
    INLIER_MEAN = OFFSET + DISTANCE_FROM_ZERO
    INLIER_STD = 0.5
    
    OUTLIER_MEAN = OFFSET - DISTANCE_FROM_ZERO
    OUTLIER_STD = 0.5
    
    
    if useTruncated:
        trueInlierSamples = getTruncatedNormSamples(0.0, numpy.inf, INLIER_MEAN, INLIER_STD, NR_INLIER_SAMPLES)
        trueOutlierSamples = getTruncatedNormSamples(-numpy.inf, 0.0, OUTLIER_MEAN, OUTLIER_STD, NR_OUTLIER_SAMPLES)
        # trueInlierSamples = scipy.stats.truncnorm.rvs(0.0, numpy.inf, loc=INLIER_MEAN, scale=INLIER_STD, size=NR_INLIER_SAMPLES)
        # trueOutlierSamples = scipy.stats.truncnorm.rvs(-numpy.inf, 0.0, loc=OUTLIER_MEAN, scale=OUTLIER_STD, size=NR_OUTLIER_SAMPLES)
    else:
        trueInlierSamples = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_INLIER_SAMPLES)
        trueOutlierSamples = scipy.stats.norm.rvs(loc=OUTLIER_MEAN, scale=OUTLIER_STD, size=NR_OUTLIER_SAMPLES)
    
    allSamples = numpy.hstack((trueOutlierSamples, trueInlierSamples)) 
    trueOutlierIndices = numpy.zeros(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES, dtype = numpy.int)
    trueOutlierIndices[0:NR_OUTLIER_SAMPLES] = 1
    
    # shuffle samples
    rndPos = numpy.arange(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES)
    numpy.random.shuffle(rndPos)
    allSamples = allSamples[rndPos]
    trueOutlierIndices = trueOutlierIndices[rndPos]
    
    return allSamples.astype(numpy.float32), trueOutlierIndices, trueInlierSamples.astype(numpy.float32), trueOutlierSamples.astype(numpy.float32)


def generateDataAdvanced_2outlierGroups(NR_SAMPLES, OUTLIER_RATIO):
    assert(OUTLIER_RATIO >= 0.0 and OUTLIER_RATIO <= 0.25)
    numpy.random.seed(3523421)
    
    NR_OUTLIER_SAMPLES = int(NR_SAMPLES * OUTLIER_RATIO)
    NR_INLIER_SAMPLES = NR_SAMPLES - NR_OUTLIER_SAMPLES
    
    # DISTANCE_FROM_ZERO = 1000.5
    # DISTANCE_FROM_ZERO = 1.5
    DISTANCE_FROM_ZERO = 2.0
    
    EXTRA_OUTLIER = -50.0
    OFFSET = 0.0
    
    INLIER_MEAN = OFFSET + DISTANCE_FROM_ZERO
    INLIER_STD = 1.0
    
    OUTLIER_MEAN = OFFSET - DISTANCE_FROM_ZERO
    OUTLIER_STD = 0.5
    
    
    trueInlierSamples = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_INLIER_SAMPLES)
    trueOutlierSamples = scipy.stats.norm.rvs(loc=OUTLIER_MEAN, scale=OUTLIER_STD, size=NR_OUTLIER_SAMPLES-3)
    
    allSamples = numpy.hstack(([EXTRA_OUTLIER, EXTRA_OUTLIER-2, EXTRA_OUTLIER-3], trueOutlierSamples, trueInlierSamples))
    trueOutlierIndices = numpy.zeros(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES, dtype = numpy.int)
    trueOutlierIndices[0:NR_OUTLIER_SAMPLES] = 1
    
    # shuffle samples
    rndPos = numpy.arange(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES)
    numpy.random.shuffle(rndPos)
    allSamples = allSamples[rndPos]
    trueOutlierIndices = trueOutlierIndices[rndPos]
    
    assert(allSamples.shape[0] == NR_SAMPLES)
    return INLIER_MEAN, INLIER_STD, allSamples.astype(numpy.float32), trueOutlierIndices, trueInlierSamples.astype(numpy.float32), trueOutlierSamples.astype(numpy.float32)


# nrOutliers, rndOutlierIds,trueOutlierIds_zeroOne = getNrOutlierAndRndOutlierIds(y_train.shape[0], TRAINING_DATA_OUTLIER_RATIO)

def getNrOutlierAndRndOutlierIds(n, OUTLIER_RATIO):
    nrOutliers = int(n * OUTLIER_RATIO)
    trueOutlierIds = numpy.arange(n)
    numpy.random.shuffle(trueOutlierIds)
    trueOutlierIds = trueOutlierIds[0:nrOutliers]
    
    trueOutlierIds_zeroOne = numpy.zeros(n, dtype = numpy.int)
    trueOutlierIds_zeroOne[trueOutlierIds] = 1
    return nrOutliers, trueOutlierIds, trueOutlierIds_zeroOne


        
def addUniformNoise(y, nrOutliers, rndOutlierIds, symmetric):
    if nrOutliers == 0:
        return y
    
    y_std = numpy.std(y)
    CUT_OFFSET = 3.0 * y_std
    OUTLIER_LENGTH = 12.0 * y_std
    
    trueOutlierSamplesRaw = numpy.random.uniform(low=0.0, high=OUTLIER_LENGTH, size=nrOutliers)
    
    if symmetric:
        lowerOutliers = -trueOutlierSamplesRaw[trueOutlierSamplesRaw < OUTLIER_LENGTH/2] - CUT_OFFSET 
        higherOutliers = trueOutlierSamplesRaw[trueOutlierSamplesRaw >= OUTLIER_LENGTH/2] - (OUTLIER_LENGTH/2) + CUT_OFFSET
    else:
        # scale by 0.5 in order to make symmetric and unsymmetric equally difficult
        lowerOutliers = -trueOutlierSamplesRaw * 0.5 - CUT_OFFSET
        higherOutliers = []
        if numpy.random.uniform() > 0.5:
            # swap
            higherOutliers = -1.0 * lowerOutliers
            lowerOutliers = []

    # print("lowerOutliers = ", lowerOutliers)
    # print("higherOutliers = ", higherOutliers)
    
    noise = numpy.hstack((lowerOutliers, higherOutliers))
    assert(noise.shape[0] == rndOutlierIds.shape[0])
    y[rndOutlierIds] += noise
    return y

def addTruncatedNormalNoise(y, nrOutliers, rndOutlierIds):
    if nrOutliers == 0:
        return y

    y_std = numpy.std(y)
    CUT_OFFSET = 3.0 * y_std

    STANDARD_LOC = 0.0
    STANDARD_SCALE = 5.0 * y_std
    
    noise = scipy.stats.truncnorm.rvs(a = (CUT_OFFSET - STANDARD_LOC) / STANDARD_SCALE, b = numpy.inf, loc = STANDARD_LOC, scale = STANDARD_SCALE, size = nrOutliers)
    signs = -2.0 * numpy.random.randint(low = 0, high = 2, size = nrOutliers) + 1.0

    noise = noise * signs

    assert(noise.shape[0] == rndOutlierIds.shape[0])
    y[rndOutlierIds] += noise
    return y


def generateDataAdvanced_UniformOutliers(NR_SAMPLES, OUTLIER_RATIO):
    assert(OUTLIER_RATIO >= 0.0 and OUTLIER_RATIO <= 0.5)
    
    INLIER_MEAN = 0.0
    INLIER_STD = 1.0
    
    y = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_SAMPLES)
    nrOutliers, rndOutlierIds, trueOutlierIds_zeroOne = getNrOutlierAndRndOutlierIds(NR_SAMPLES, OUTLIER_RATIO)
    
    y = addUniformNoise(y, nrOutliers, rndOutlierIds, symmetric = True)
    
    return INLIER_MEAN, INLIER_STD, y.astype(numpy.float32), trueOutlierIds_zeroOne


def generateDataAdvanced_UniformOutliers_old(NR_SAMPLES, OUTLIER_RATIO):
    assert(OUTLIER_RATIO >= 0.0 and OUTLIER_RATIO < 0.5)
    
    NR_OUTLIER_SAMPLES = int(NR_SAMPLES * OUTLIER_RATIO)
    NR_INLIER_SAMPLES = NR_SAMPLES - NR_OUTLIER_SAMPLES
    
    
    INLIER_MEAN = 0.0
    INLIER_STD = 1.0
    
    OUTLIER_LENGTH = 12.0
    
    trueInlierSamples = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_INLIER_SAMPLES)
    trueOutlierIndices = numpy.zeros(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES, dtype = numpy.int)    
    
    if OUTLIER_RATIO > 0.0:
        trueOutlierSamplesRaw = numpy.random.uniform(low=0.0, high=OUTLIER_LENGTH, size=NR_OUTLIER_SAMPLES)
        lowerOutliers = -trueOutlierSamplesRaw[trueOutlierSamplesRaw < OUTLIER_LENGTH/2] - (INLIER_STD*3.0)
        higherOutliers = trueOutlierSamplesRaw[trueOutlierSamplesRaw >= OUTLIER_LENGTH/2] - (OUTLIER_LENGTH/2) + (INLIER_STD*3.0)
        
        trueOutlierSamples = numpy.hstack((lowerOutliers, higherOutliers))
        assert(trueOutlierSamples.shape[0] == NR_OUTLIER_SAMPLES)
        allSamples = numpy.hstack((trueOutlierSamples, trueInlierSamples))
        trueOutlierIndices[0:NR_OUTLIER_SAMPLES] = 1
        
        # shuffle samples
        rndPos = numpy.arange(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES)
        numpy.random.shuffle(rndPos)
        allSamples = allSamples[rndPos]
        trueOutlierIndices = trueOutlierIndices[rndPos]
        
    else:
        allSamples = numpy.copy(trueInlierSamples)
        trueOutlierSamples = numpy.zeros(0)
        
    assert(allSamples.shape[0] == NR_SAMPLES)
    return INLIER_MEAN, INLIER_STD, allSamples.astype(numpy.float32), trueOutlierIndices, trueInlierSamples.astype(numpy.float32), trueOutlierSamples.astype(numpy.float32)


def generateDataAdvanced_noOutlier(NR_SAMPLES, INLIER_STD):
    numpy.random.seed(3523421)
         
    INLIER_MEAN = 0.0
    
    trueInlierSamples = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_SAMPLES)
    trueOutlierSamples = scipy.stats.norm.rvs(loc=0.0, scale=1.0, size=0)
    
    allSamples = numpy.hstack((trueOutlierSamples, trueInlierSamples))
    trueOutlierIndices = numpy.zeros(NR_SAMPLES, dtype = numpy.int)
    
    assert(allSamples.shape[0] == NR_SAMPLES)
    return INLIER_MEAN, INLIER_STD, allSamples.astype(numpy.float32), trueOutlierIndices, trueInlierSamples.astype(numpy.float32), trueOutlierSamples.astype(numpy.float32)


def generateDataAdvanced_2outlierGroupsBreakHardSort(NR_SAMPLES, OUTLIER_RATIO):
    assert(OUTLIER_RATIO >= 0.0 and OUTLIER_RATIO <= 0.25)
    numpy.random.seed(3523421)
    
    NR_OUTLIER_SAMPLES = int(NR_SAMPLES * OUTLIER_RATIO)
    NR_INLIER_SAMPLES = NR_SAMPLES - NR_OUTLIER_SAMPLES
        
    OFFSET = 10.0
    
    INLIER_MEAN = OFFSET
    INLIER_STD = 3.0
    
    OUTLIER_MEAN = 0.0
    OUTLIER_STD = 0.5
    
    trueInlierSamples = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_INLIER_SAMPLES)
    trueOutlierSamples = scipy.stats.norm.rvs(loc=OUTLIER_MEAN, scale=OUTLIER_STD, size=NR_OUTLIER_SAMPLES)
    
    # allSamples = numpy.hstack(([EXTRA_OUTLIER], trueOutlierSamples, trueInlierSamples))
    allSamples = numpy.hstack((trueOutlierSamples, trueInlierSamples))
    trueOutlierIndices = numpy.zeros(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES, dtype = numpy.int)
    trueOutlierIndices[0:NR_OUTLIER_SAMPLES] = 1
    
    # shuffle samples
    rndPos = numpy.arange(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES)
    numpy.random.shuffle(rndPos)
    allSamples = allSamples[rndPos]
    trueOutlierIndices = trueOutlierIndices[rndPos]
    
    assert(allSamples.shape[0] == NR_SAMPLES)
    return INLIER_MEAN, INLIER_STD, allSamples.astype(numpy.float32), trueOutlierIndices, trueInlierSamples.astype(numpy.float32), trueOutlierSamples.astype(numpy.float32)


def generateDataAdvanced_largeVarianceSeparated(NR_SAMPLES, OUTLIER_RATIO, offset):
    assert(OUTLIER_RATIO >= 0.0 and OUTLIER_RATIO <= 0.25)
    numpy.random.seed(3523421)
    
    NR_OUTLIER_SAMPLES = int(NR_SAMPLES * OUTLIER_RATIO)
    NR_INLIER_SAMPLES = NR_SAMPLES - NR_OUTLIER_SAMPLES
        
    
    INLIER_MEAN = 0.0
    INLIER_STD = 3.0
    
    OUTLIER_MEAN = offset
    OUTLIER_STD = 0.5
    
    trueInlierSamples = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_INLIER_SAMPLES)
    trueOutlierSamples = scipy.stats.norm.rvs(loc=OUTLIER_MEAN, scale=OUTLIER_STD, size=NR_OUTLIER_SAMPLES)
    
    # allSamples = numpy.hstack(([EXTRA_OUTLIER], trueOutlierSamples, trueInlierSamples))
    allSamples = numpy.hstack((trueOutlierSamples, trueInlierSamples))
    trueOutlierIndices = numpy.zeros(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES, dtype = numpy.int)
    trueOutlierIndices[0:NR_OUTLIER_SAMPLES] = 1
    
    # shuffle samples
    rndPos = numpy.arange(NR_OUTLIER_SAMPLES + NR_INLIER_SAMPLES)
    numpy.random.shuffle(rndPos)
    allSamples = allSamples[rndPos]
    trueOutlierIndices = trueOutlierIndices[rndPos]
    
    assert(allSamples.shape[0] == NR_SAMPLES)
    return INLIER_MEAN, INLIER_STD, allSamples.astype(numpy.float32), trueOutlierIndices, trueInlierSamples.astype(numpy.float32), trueOutlierSamples.astype(numpy.float32)



def generateDataAdvanced_breakHardSortReally(NR_SAMPLES, OUTLIER_RATIO):
    numpy.random.seed(3523421)
    
    NR_OUTLIER_SAMPLES = int(NR_SAMPLES * OUTLIER_RATIO)
    NR_INLIER_SAMPLES = NR_SAMPLES - NR_OUTLIER_SAMPLES
    
    NR_OUTLIER_SAMPLES_GROUP1 = int(NR_OUTLIER_SAMPLES / 2)
    NR_OUTLIER_SAMPLES_GROUP2 = NR_OUTLIER_SAMPLES - NR_OUTLIER_SAMPLES_GROUP1
    assert(NR_SAMPLES == NR_OUTLIER_SAMPLES_GROUP1 + NR_OUTLIER_SAMPLES_GROUP2 + NR_INLIER_SAMPLES)
    
    OFFSET = 10.0
    
    INLIER_MEAN = OFFSET + 5.0
    INLIER_STD = 1.0
    
    OUTLIER_MEAN1 = OFFSET
    OUTLIER_STD1 = 1.0
    
    OUTLIER_MEAN2 = 0.0
    OUTLIER_STD2 = 1.0
    
    trueInlierSamples = scipy.stats.norm.rvs(loc=INLIER_MEAN, scale=INLIER_STD, size=NR_INLIER_SAMPLES)
    trueOutlierSamples1 = scipy.stats.norm.rvs(loc=OUTLIER_MEAN1, scale=OUTLIER_STD1, size=NR_OUTLIER_SAMPLES_GROUP1)
    trueOutlierSamples2 = scipy.stats.norm.rvs(loc=OUTLIER_MEAN2, scale=OUTLIER_STD2, size=NR_OUTLIER_SAMPLES_GROUP2)
    
    trueOutlierSamples = numpy.hstack((trueOutlierSamples1, trueOutlierSamples2))
    allSamples = numpy.hstack((trueOutlierSamples, trueInlierSamples))
    trueOutlierIndices = numpy.zeros(NR_SAMPLES, dtype = numpy.int)
    trueOutlierIndices[0:NR_OUTLIER_SAMPLES] = 1
    assert(trueOutlierSamples.shape[0] == NR_OUTLIER_SAMPLES)
    
    # shuffle samples
    rndPos = numpy.arange(NR_SAMPLES)
    numpy.random.shuffle(rndPos)
    allSamples = allSamples[rndPos]
    trueOutlierIndices = trueOutlierIndices[rndPos]
    
    assert(allSamples.shape[0] == NR_SAMPLES)
    return INLIER_MEAN, INLIER_STD, allSamples.astype(numpy.float32), trueOutlierIndices, trueInlierSamples.astype(numpy.float32), trueOutlierSamples.astype(numpy.float32)


def addNoise_and_scale(X_train, y_train, X_cleanTest, y_cleanTest, TRAINING_DATA_OUTLIER_RATIO, noiseType):

    NORMALIZE_DATA_X = True
    NORMALIZE_DATA_Y = True

    # ***************************************************
    nrOutliers, trueOutlierIds, trueOutlierIds_zeroOne = getNrOutlierAndRndOutlierIds(y_train.shape[0], TRAINING_DATA_OUTLIER_RATIO)

    if noiseType == "noNoise":
        assert(TRAINING_DATA_OUTLIER_RATIO == 0.0)
        nrOutliers = 0
    elif noiseType == "normalNoise":
        assert(TRAINING_DATA_OUTLIER_RATIO > 0.0)
        # add outliers as in "Fast Differentiable Sorting and Ranking"
        y_train_std = numpy.std(y_train)
        noise = scipy.stats.norm.rvs(loc = 0.0, scale = 5.0 * y_train_std, size = nrOutliers)
        y_train[trueOutlierIds] += noise
    elif noiseType == "normalNoisePlusBigOutlier":
        assert(TRAINING_DATA_OUTLIER_RATIO > 0.0)
        y_train_std = numpy.std(y_train)
        noise = scipy.stats.norm.rvs(loc = 0.0, scale = 5.0 * y_train_std, size = nrOutliers)
        y_train[trueOutlierIds] += noise
        y_train[trueOutlierIds[0]] += 100.0 * y_train_std

    elif noiseType == "normalNoisePlusBigOutlier_truncated":
        assert(TRAINING_DATA_OUTLIER_RATIO > 0.0)
        y_train = addTruncatedNormalNoise(y_train, nrOutliers, trueOutlierIds)
        y_train[trueOutlierIds[0]] += 100.0 * numpy.std(y_train)

    elif noiseType == "inputNormalNoise":
        assert(TRAINING_DATA_OUTLIER_RATIO > 0.0)
        x_train_std = numpy.std(X_train,axis = 0)
        
        for outlierId in range(nrOutliers):
            noiseVec = scipy.stats.norm.rvs(loc = numpy.zeros_like(x_train_std), scale = 5.0 * x_train_std)
            X_train[trueOutlierIds[outlierId], :] += noiseVec
    elif noiseType == "uniform":
        y_train = addUniformNoise(y_train, nrOutliers, trueOutlierIds, symmetric = True)
    elif noiseType == "asymmetric":
        y_train = addUniformNoise(y_train, nrOutliers, trueOutlierIds, symmetric = False)
    elif noiseType == "focused":
        assert(TRAINING_DATA_OUTLIER_RATIO > 0.0)

        d = X_train.shape[1]
        
        midPoint = int(X_train.shape[0] / 2)

        FOCUS_POINTS_IDS = numpy.argsort(X_train, axis = 0)[midPoint, :]

        CONCENTRATION_X = scipy.stats.median_abs_deviation(X_train, axis = 0) * 0.1 * d
        jitterX = CONCENTRATION_X * (numpy.random.uniform(low=0.0, high=1.0, size=(nrOutliers, d)) - 0.5)

        CONCENTRATION_Y = scipy.stats.median_abs_deviation(y_train) * 0.1
        OFFSET_Y = 3.0 * numpy.std(y_train)
        jitterY = CONCENTRATION_Y * numpy.random.uniform(low=0.0, high=1.0, size=nrOutliers)
        
        X_train[trueOutlierIds, :] = X_train[FOCUS_POINTS_IDS, numpy.arange(d)] + jitterX       # median of each dimension + jitter
        y_train[trueOutlierIds] = numpy.median(y_train[FOCUS_POINTS_IDS]) - OFFSET_Y - jitterY   # median reponse - offset - jitter


    elif noiseType == "FriedmanNoise1":
        # as in "Robust Regression with twinned Gaussian Processes", page 7, (Friedman (1)), uses n = 100
        assert(TRAINING_DATA_OUTLIER_RATIO == 0.1)
        nrOutliers = numpy.sum(trueOutlierIds_zeroOne)
        assert(nrOutliers == 10)
        y_train[trueOutlierIds] = numpy.random.normal(loc = 15, scale = 3, size = nrOutliers) 

    elif noiseType == "FriedmanNoise2":
        # as in "Robust Regression with twinned Gaussian Processes", page 7, (Friedman (2)), uses n = 100
        assert(TRAINING_DATA_OUTLIER_RATIO == 0.1)
        assert(X.shape[1] == 10)
        nrOutliers = numpy.sum(trueOutlierIds_zeroOne)
        y_train[trueOutlierIds] = numpy.random.normal(loc = 0, scale = 1, size = nrOutliers) 
        # print("y_train[trueOutlierIds] = ", y_train[trueOutlierIds])
        # assert(False)
        mu1 = numpy.random.rand(X.shape[1])
        mu2 = numpy.random.rand(X.shape[1])
        X_train[trueOutlierIds[0:int(nrOutliers / 2)], :] = numpy.random.normal(loc = mu1, scale = numpy.sqrt(0.001) * numpy.ones_like(mu1), size = (int(nrOutliers / 2), X.shape[1]))
        X_train[trueOutlierIds[int(nrOutliers / 2):nrOutliers], :] = numpy.random.normal(loc = mu2, scale = numpy.sqrt(0.001) * numpy.ones_like(mu2), size = (int(nrOutliers / 2), X.shape[1]))
    else:
        assert(False)
    # **************************************************
    
    if NORMALIZE_DATA_X:
        X_train, X_cleanTest, dataScalerX = commons_data_preparation.scale_X(X_train, X_cleanTest)
    else:
        dataScalerX = None

    
    if NORMALIZE_DATA_Y:
        y_train, y_cleanTest, dataScalerY = commons_data_preparation.scale_y(y_train, y_cleanTest)
    else:
        dataScalerY = None
    
    
    return X_train, y_train, trueOutlierIds_zeroOne, X_cleanTest, y_cleanTest, dataScalerX, dataScalerY
