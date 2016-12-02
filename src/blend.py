"""
BSD License, 3 clauses.

"""
import numpy as np
import pandas as pd
import load_data
import nn
import encode
import xgboost as xgb
import cPickle as pickle

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

    n_folds = 10
    shuffle = False

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
    add_feats = False 
    if (add_feats):
        X = np.append(X, X_n, axis=1)
        X_submission = np.append(X_submission, X_submission_n, axis=1)
    else: # add data to the training dataset
        X_n, X_submission_n = nn.load_data_for_decoder()
        X_n = encode.fit_transform(X_n)
        # add data by fit transform the orginal into same attributes data and then add to the data set
        X = np.append(X, X_n, axis=0)
        y = np.append(y, y, axis=0)
        

 

    # if shuffle:
        # idx = np.random.permutation(y.size)
        # X = X[idx]
        # y = y[idx]
    # print X.shape
    # print y.shape

    
    skf = StratifiedKFold(y, n_folds=n_folds)
    print 'num of splits'
    print len(skf) 

    clfs = [ 
            Lasso(alpha=0.001, max_iter=50000),
            # LassoCV(alphas = [1,0.1, 0.001, 0.0005, 5e-4], max_iter=50000),
            # LassoCV(alphas = [1,0.1, 0.001, 0.0005, 5e-4], max_iter=50000, cv=10),
            ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=50000),
            # ElasticNetCV(alphas=[1,0.1,0.001,0.0005,5e-4], l1_ratio=0.5, max_iter=50000),
            # xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=2, min_child_weight=3, gamma=0.0, subsample=0.6, colsample_bytree=0.6),
            # BayesianRidge(n_iter=1000),
            # ExtraTreesRegressor(n_estimators=380, max_depth=60),
            # RandomForestRegressor(n_estimators=400, max_depth=50)
            ]

    print "Creating train and test sets for blending."

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    model_error = np.zeros(len(clfs))


    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        model_error_j = np.zeros(len(skf))
        for i, (train_ind, test_ind) in enumerate(skf):
            print "Fold", i
            X_train = X[train_ind]
            y_train = y[train_ind]
            X_test = X[test_ind]
            y_test = y[test_ind]
            clf.fit(X_train, y_train)
            y_submission = clf.predict(X_test)
            print mean_squared_error(y_submission, y_test)
            model_error_j[i] = mean_squared_error(y_submission, y_test)
            dataset_blend_train[test_ind, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict(X_submission)
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        model_error[j] = model_error_j.mean()

    # print
    # print "Blending."
    # clf = LinearRegression()
    # clf.fit(dataset_blend_train, y)
    # y_submission = clf.predict(dataset_blend_test)
    
    # print "rescale to original one"
    # y_submission = np.expm1(y_submission)
   
    # print "Saving Results."
    # solution = pd.DataFrame({"id":Id, "SalePrice":y_submission})
    # solution.to_csv("stacking.csv", index= False)


    print "error {}".format(model_error)
    model_error = 1./model_error 
    inverse_model_error = model_error / sum(model_error)
    print "weights of models {} (inverse error)".format(inverse_model_error)

    print "rescale to original one"
    dataset_blend_test = np.expm1(dataset_blend_test)

    y_submission = np.sum(dataset_blend_test * inverse_model_error, axis=1)

    # print "Saving Results."
    # solution = pd.DataFrame({"id":Id, "SalePrice":y_submission})
    # solution.to_csv("weighted_blend.csv", index= False)

