import pandas as pd
import numpy as np
from scipy.stats import skew





def load():
    train = pd.read_csv("../input/train.csv", error_bad_lines=False)
    test = pd.read_csv("../input/test.csv",error_bad_lines=False)
# remove outlier
    train.drop(train[train["GrLivArea"]>4000].index, inplace=True)
    train.drop(train[train["EnclosedPorch"]>500].index, inplace=True)


    quality_feats = ["MSZoning","OverallQual","OverallCond","Neighborhood","ExterQual", "ExterCond", "BsmtQual","BsmtCond","HeatingQC", "KitchenQual","FireplaceQu","GarageQual","GarageCond","BsmtExposure", "BsmtFinType1","BsmtFinType2", "Functional", "Fence", "GarageFinish", "PavedDrive"]

    qual_dict = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    bsmt_fin_dict = {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    bsmt_exposure_dict = {"No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    Functional_dict = {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                         "Min2": 6, "Min1": 7, "Typ": 8}
    Fence_dict = {"MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}
    GarageFinish_dict = {"Unf": 1, "RFn": 2, "Fin": 3}

    pave_drive_dict = {"Y": 3, "P":2, "N": 1}

    for feature in quality_feats:
#     dic = train["SalePrice"].groupby(train[feature]).mean().sort_values()
        dic ={}
        if feature in ["Neighborhood", "OverallQual", "OverallCond", "MSZoning"]:
            dic = train["SalePrice"].groupby(train[feature]).mean().sort_values()
        elif feature is "BsmtExposure":
            dic = bsmt_exposure_dict
        elif feature in ["BsmtFinType1", "BsmtFinType2"]:
            dic = bsmt_fin_dict
        elif feature is "Functional":
            dic = Functional_dict
        elif feature is "Fence":
            dic = Fence_dict
        elif feature is "GarageFinish":
            dic = GarageFinish_dict
        elif feature is "PavedDrive":
            dic = pave_drive_dict
        else: dic  = qual_dict
        train.insert(1, "_"+feature, train[feature].replace(dic))
        test.insert(1, "_"+feature, test[feature].replace(dic))
        # train[feature] = train[feature].replace(dic)
        # test[feature] = test[feature].replace(dic)
    # print train.columns
    test.loc[666, "GarageQual"] = "TA"
    test.loc[666, "GarageCond"] = "TA"
    test.loc[666, "GarageFinish"] = "Unf"
    test.loc[666, "GarageYrBlt"] = "1980"
    train.insert(1, "HouseAge", 2010 - train["YearBuilt"])
    test.insert(1, "HouseAge", 2010 - test["YearBuilt"])
    train.insert(1, "TimeSinceSold", 2010 - train["YrSold"])
    test.insert(1, "TimeSinceSold", 2010 - test["YrSold"])



    all_data = pd.concat((train.loc[:,"TimeSinceSold":'SaleCondition'],
                          test.loc[:,"TimeSinceSold":'SaleCondition']))

    all_data["GarageYrBlt"] = pd.to_numeric(all_data["GarageYrBlt"], errors='ignore')

    # feature data replacing
    all_data = all_data.replace({'MoSold': {1: 'Jan', 
                                        2: 'Feb', 
                                        3: 'Mar', 
                                        4: 'Apr',
                                        5: 'May', 
                                        6: 'Jun',
                                        7: 'Jul',
                                        8: 'Avg',
                                        9: 'Sep',
                                        10: 'Oct',
                                        11: 'Nov',
                                        12: 'Dec'}})
    # log transform the target:
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) # compute skewness
    skewed_feats = skewed_feats[skewed_feats>0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data = all_data.fillna({'MasVnrType': 'None'})

    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())

    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice
    
    # drop columns that only exist less than 4 value in test data to avoid
    # overfitting
    drop_columns = X_test.columns[(X_test.astype(bool).sum(axis=0) < 5)]
    X_train.drop(drop_columns, axis=1, inplace=True)
    X_test.drop(drop_columns, axis=1, inplace=True)

    print X_train.shape
    print y.shape
    print X_test.shape
    return X_train, y, X_test, test.Id
