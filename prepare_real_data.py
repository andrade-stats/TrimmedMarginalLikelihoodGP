
import pandas as pd
import numpy
import commons_data_preparation
import commonSettings

# TopGear data has been analyzed in:
# "Robust groupwise least angle regression.", Alfons et al, Computational Statistics & Data Analysis, 2016
# "Sparse regression for large data sets with outliers", Bottmer et al, European Journal of Operational Research, 2022
# data it self is from R-Package "robustHD: Robust Methods for High-Dimensional Data"
def prepareTopGear_data():

    datasetName = "TopGear"
    
    BASE_FOLDER = "/Users/danielandrade/ResearchProjects/TrimmedLikelihoodOD_withGP/openDatasets/"

    df_all = pd.read_csv(BASE_FOLDER + datasetName + ".csv")
        
    X = df_all[["Price", "Displacement", "BHP", "Torque", "Acceleration", "TopSpeed", "Weight", "Length", "Width", "Height"]].to_numpy()
    y = df_all["MPG"].to_numpy()

    # ****************** ignore entries with NAN ****************

    nan_ids_X = numpy.any(numpy.isnan(X), axis = 1)
    nan_ids_y = numpy.isnan(y)
    assert(nan_ids_y.shape[0] == nan_ids_X.shape[0])
    nan_ids = numpy.logical_or(nan_ids_X, nan_ids_y)
    
    X = X[numpy.logical_not(nan_ids), :]
    y = y[numpy.logical_not(nan_ids)]

    print("number of samples with one or more nan (y) = ", numpy.sum(nan_ids_y))
    print("number of samples with one or more nan (total) = ", numpy.sum(nan_ids))

    # ****************** transform y ****************

    y = numpy.log(y) # use log(MPG) as in "Sparse regression for large data sets with outliers"

    # ***************** save unscaled data ************

    allData_original = {}
    allData_original["X_original"] = X
    allData_original["y_original"] = y
    numpy.save("openDatasets/" + datasetName,  allData_original)

    # ****************** scale X and y ****************
    
    X_train, X_test, dataScalerX = commons_data_preparation.scale_X(X)
    y_train, y_test, dataScalerY = commons_data_preparation.scale_y(y)

    
    # ****************** save  ****************

    all_trueOutlierIndicesZeroOne = [numpy.zeros(y_train.shape[0], dtype = numpy.int64)]
    DATA_SPLIT = "wholeData"
    
    save_no_noise_data(X_train, y_train, X_test, y_test, dataScalerX, dataScalerY, datasetName, all_trueOutlierIndicesZeroOne, DATA_SPLIT)
    return



def prepareDengue_data():
    BASE_FOLDER = "/Users/danielandrade/ResearchProjects/TrimmedLikelihoodOD_withGP/openDatasets/dengue-forecasting-project-2015-master/Dengue_data/"

    # LOCATION = "san_juan"
    LOCATION = "iquitos"

    TRAINING_DATA_FILENAME = BASE_FOLDER + LOCATION + "_training_data.csv"
    TEST_DATA_FILENAME = BASE_FOLDER + LOCATION + "_testing_data.csv"

    df_training = pd.read_csv(TRAINING_DATA_FILENAME)
    df_all = pd.read_csv(TEST_DATA_FILENAME)

    df_all.insert(0, "sine_wave", 0)
    df_all.insert(0, "starting_level", 0)

    print(df_all)

    # ****************** add Sine wave ******************
    for i in range(len(df_all)):
        season_week = df_all.loc[i, "season_week"] - 1
        assert(season_week <= 51)
        df_all.loc[i,"sine_wave"] = numpy.sin(numpy.pi * (season_week / 52))

    # ****************** square root transformation for total_cases ******************
    for i in range(len(df_all)):
        df_all.loc[i, "total_cases"] = numpy.sqrt(df_all.loc[i, "total_cases"])

    # ****************** add Starting level ******************
    for i in range(1, len(df_all)):
        if i == 1 or df_all.loc[i, "season_week"] == 1:
            cases_previous_season = df_all.loc[i - 1, "total_cases"]
            print("cases_previous_season = ", cases_previous_season)
        df_all.loc[i,"starting_level"] = cases_previous_season

    # ****************** remove first row (since we have no starting_level for that one) ******************
    df_all = df_all.drop(0)


    # pd.set_option('display.max_rows', None)
    # print(df_all)


    # ****************** split into train and test data
    df_train = df_all.loc[df_all["week_start_date"].isin(df_training["week_start_date"])]
    df_test = df_all.copy()

    for date in df_training["week_start_date"]:
        df_test.drop(df_test[df_test["week_start_date"] == date].index, inplace=True)
        
    # print("df_train = ", df_train)
    # print("df_test = ", df_test)


    # ****************** get numpy arrays and scale X and y ****************
    def getNumpyArrays(df):
        X = df[["season_week", "sine_wave", "starting_level"]].to_numpy()
        y = df["total_cases"].to_numpy()
        return X, y

    X_train, y_train = getNumpyArrays(df_train)
    X_test, y_test = getNumpyArrays(df_test)

    X_train, X_test, dataScalerX = commons_data_preparation.scale_X(X_train, X_test)
    y_train, y_test, dataScalerY = commons_data_preparation.scale_y(y_train, y_test)

    # ****************** save  ****************

    datasetName = "dengue" + "_" + LOCATION
    all_trueOutlierIndicesZeroOne = [numpy.zeros(y_train.shape[0], dtype = numpy.int64)]
    DATA_SPLIT = "trainTestData"

    save_no_noise_data(X_train, y_train, X_test, y_test, dataScalerX, dataScalerY, datasetName, all_trueOutlierIndicesZeroOne, DATA_SPLIT)
    return


def save_no_noise_data(X_train, y_train, X_test, y_test, dataScalerX, dataScalerY, datasetName, all_trueOutlierIndicesZeroOne, DATA_SPLIT):
    assert(DATA_SPLIT == "wholeData" or DATA_SPLIT == "trainTestData")

    print("X_train = ", X_train.shape)
    print("y_train = ", y_train.shape)
    print("X_test = ", X_test.shape)
    print("y_test = ", y_test.shape)

    all_X_train = [X_train]
    all_y_train = [y_train]
    all_X_cleanTest = [X_test]
    all_y_cleanTest = [y_test]
    all_dataScalerX = [dataScalerX]
    all_dataScalerY = [dataScalerY]

    allData = {}
    allData["all_X_train"] = all_X_train
    allData["all_y_train"] = all_y_train
    allData["all_trueOutlierIndicesZeroOne"] = all_trueOutlierIndicesZeroOne
    allData["all_dataScalerX"] = all_dataScalerX
    allData["all_dataScalerY"] = all_dataScalerY

    allData["all_X_cleanTest"] = all_X_cleanTest
    allData["all_y_cleanTest"] = all_y_cleanTest

    NOISE_TYPE = "noNoise"
    _, noisePostFix = get_noise_info(NOISE_TYPE)
    numpy.save(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + DATA_SPLIT + "_" + NOISE_TYPE + noisePostFix,  allData)

    print("SAVED SUCCESFULLY DATA ", datasetName)

def get_noise_info(NOISE_TYPE):

    if NOISE_TYPE == "noNoise":
        TRUE_OUTLIER_RATIO = 0.0
        noisePostFix = ""
    else:
        assert(False)

    return TRUE_OUTLIER_RATIO, noisePostFix


prepareTopGear_data()
