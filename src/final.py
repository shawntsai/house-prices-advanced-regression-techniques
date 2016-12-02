"""
BSD License, 3 clauses.

"""
import numpy as np
import cPickle as pickle
import nn
import encode

import pandas as pd
import load_data
import xgboost as xgb
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, BayesianRidge, LinearRegression, Lasso, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor



def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return rmse

if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set

    try:
        X_n, X_submission_n= pickle.load(open("X_n.pkl", "rb"))
    except (OSError, IOError) as e:
        X_n, X_submission_n = nn.load_data_for_decoder()

        # X_n = encode.fit_transform(X_n)
        # X_submission_n = encode.fit_transform(X_submission_n)
        X_n = encode.get_encoded_feature(X_n)
        X_submission_n = encode.get_encoded_feature(X_submission_n)

        pickle.dump((X_n, X_submission_n), open("X_n.pkl", "wb"))


    X, y, X_submission, Id = load_data.load()

    X, X_submission = X.values, X_submission.values

    y=y.values
    # include auto-encode features
    add_feats = True
    if (add_feats):
        X = np.append(X, X_n, axis=1)
        X_submission = np.append(X_submission, X_submission_n, axis=1)
    else: # add data to the training dataset
        X_n, X_submission_n = nn.load_data_for_decoder()
        X_n = encode.fit_transform(X_n)
        # add data by fit transform the orginal into same attributes data and then add to the data set
        X = np.append(X, X_n, axis=0)
        y = np.append(y, y, axis=0)
    print X.shape
        


    clfs = [ 
            Lasso(alpha=0.001, max_iter=50000),
            LassoCV(alphas = [1,0.1, 0.001, 0.0005, 5e-4], max_iter=50000, cv=10) ,
            # LassoCV(alphas = [1,0.1, 0.001, 0.0005, 5e-4], max_iter=50000) ,
            ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=50000),
            # ElasticNetCV(alphas=[1,0.1,0.001,0.0005,5e-4], l1_ratio=0.5, max_iter=50000),
            xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=2, min_child_weight=3, gamma=0.0, subsample=0.6, colsample_bytree=0.6),

            # BayesianRidge(n_iter=1000),
            # ExtraTreesRegressor(n_estimators=380, max_depth=60),
            # RandomForestRegressor(n_estimators=400, max_depth=50)
            ]

    predictions= np.zeros((X_submission.shape[0], len(clfs)))
    for j, clf in enumerate(clfs):
        print j, clf
        clf.fit(X, y)
        predictions[:, j] = clf.predict(X_submission)
    predictions= np.expm1(predictions)
    y_submission = predictions.mean(axis=1)

    print "Saving Results."
    solution = pd.DataFrame({"id":Id, "SalePrice":y_submission})
    solution.to_csv("weighted_blend.csv", index= False)

