import pandas as pd
import numpy as np
import sklearn.model_selection as models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from enum import Enum
import matplotlib.pyplot as plt


class RegressionMethod(Enum):
    SINGLE = 'single',
    TRAIN_TEST_SPLIT = 'train_test_split',
    KFOLD = 'kfold'

class RegressorType(Enum):
    DECISION_TREE = 'decision_tree',
    RANDOM_FOREST = 'random_forest',

def singleRegressorMAE(regressor, x_train, y_test):
    regressor.fit(x_train, y_test)
    return mean_absolute_error(y_test, regressor.predict(x_train))

def trainTestSplitRegressorMAE(regressor, df_features, df_main_feature, test_size = 0.5):
    (x_train, x_test, y_train, y_test) = models.train_test_split(df_features, df_main_feature, test_size = test_size)
    regressor.fit(x_train, y_train)
    return mean_absolute_error(y_test.astype(float), regressor.predict(x_test).astype(float))

def kfoldRegressorMAE(regressor, df_features, df_main_feature, n_splits = 5):
    kf = models.KFold(n_splits = n_splits)
    result = 0
    for train, test in kf.split(df_features, df_main_feature):
        x_train, x_test = df_features.iloc[train], df_features.iloc[test]
        y_train, y_test = df_main_feature.iloc[train], df_main_feature.iloc[test]

        regressor.fit(x_train, y_train)
        mae = mean_absolute_error(y_test, regressor.predict(x_test))
        if(mae > result):
            result = mae

    return mae

def getMaeValues(dataMethod: RegressionMethod, regressorType: RegressorType, X, y, depth = 5):
    depths = []
    maes = []    
    for i in range(1, depth + 1):
        match regressorType:
            case RegressorType.DECISION_TREE:
                regressor = DecisionTreeRegressor(max_depth = i)
            case RegressorType.RANDOM_FOREST:
                regressor = RandomForestRegressor(max_depth = i)
            # case RegressorType.LINEAR:
                # regressor = LinearRegression()
        #regressor = DecisionTreeRegressor(max_depth = i) if regressorType is RegressorType.DECISION_TREE else RandomForestRegressor(max_depth = i)

        mae = 0
        match dataMethod:
            case RegressionMethod.SINGLE:
                mae = singleRegressorMAE(regressor, X, y)
            case RegressionMethod.TRAIN_TEST_SPLIT:
                mae = trainTestSplitRegressorMAE(regressor, X, y, 0.2)
            case RegressionMethod.KFOLD:
                mae = kfoldRegressorMAE(regressor, X, y)

        depths.append(i)
        maes.append(mae)
    
    return maes

def predictValueByFeatures(dataFrame, features, mainFeature):
    dataFrame = dataFrame.dropna(subset = features)

    features.remove(mainFeature)

    print(dataFrame)
    #dataFrame['key'].fillna(0,inplace = True)
    #dataFrame['in_shazam_charts'].fillna(0, inplace = True)
    #print(dataFrame.isnull().sum())

    pairs =[
        # [RegressionMethod.SINGLE, RegressorType.LINEAR],
        # [RegressionMethod.TRAIN_TEST_SPLIT, RegressorType.LINEAR],
        # [RegressionMethod.KFOLD, RegressorType.LINEAR],
        [RegressionMethod.SINGLE, RegressorType.DECISION_TREE],
        [RegressionMethod.TRAIN_TEST_SPLIT, RegressorType.DECISION_TREE],
        [RegressionMethod.KFOLD, RegressorType.DECISION_TREE],
        [RegressionMethod.SINGLE, RegressorType.RANDOM_FOREST],
        [RegressionMethod.TRAIN_TEST_SPLIT, RegressorType.RANDOM_FOREST],
        [RegressionMethod.KFOLD, RegressorType.RANDOM_FOREST]
    ]

    names = []
    results = []
    depth = 5
    for pair in pairs:
        names.append(pair[0].name +' - '+ pair[1].name)
        results.append(getMaeValues(pair[0], pair[1], dataFrame[features], dataFrame[mainFeature], depth))


    #wyświtlanie słupkowe
    data = pd.DataFrame(results, names, np.arange(1, len(results[0])+1))
    data.plot(kind="bar",figsize=(15, 8))
    plt.xticks(rotation = 0, fontsize=7)
    plt.ylabel("MAE", rotation='horizontal') 
    plt.title("MAE in certains depths of regressions") 
    plt.legend(title='Depth')
    plt.show()

    #wyświetlanie liniowe
    for i in range(len(results)):
        plt.plot(np.arange(1, len(results[0])+1, 1),results[i])
        #plt.title(names[i])
        plt.legend(names)
    plt.show()

result = predictValueByFeatures(pd.read_csv('data/spotify-2023.csv', encoding='latin-1'), ['artist_count','released_year','in_apple_playlists','in_spotify_playlists','in_spotify_charts','danceability_%', 'streams'], 'streams')