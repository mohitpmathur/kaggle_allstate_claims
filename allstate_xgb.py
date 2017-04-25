'''
This code uses dataset from the Kaggle 'Allstate Claims Severity' 
competition 
https://www.kaggle.com/c/allstate-claims-severity

The code below uses xgboost. Hyperparameters were tuned using 
GridSearch.

Tuning them, I was able to get a MSA of 1135 on the validation set

'''

from __future__ import print_function
import time
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    #Read data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    test['loss'] = np.nan
    joined = pd.concat([train, test])
    print ("train shape:", train.shape)
    print ("test shape:", test.shape)
    print ("joined shape:", joined.shape)

    print ("Starting to replace values with NaN if they exist in only train or test ...")
    for column in list(train.select_dtypes(include=['object']).columns):
        if set(train[column].unique()) != set(test[column].unique()):
            #if train[column].nunique() != test[column].nunique():
            # print ("\n",column)
            set_train = set(train[column].unique())
            set_test = set(test[column].unique())
            remove_train = set_train - set_test
            remove_test = set_test - set_train
            remove = remove_train.union(remove_test)
            # print (remove_train)
            # print (remove_test)
            # print (remove)
            def filter_cat(x):
                # Remove if exist in only either train or test
                if x in remove:
                    return np.nan
                return x
            
            joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
            #print (joined[column].unique())
        joined[column] = pd.factorize(joined[column].values, sort=True)[0]
    print ("Categorical column factorized")
    
    # # One-hot-encode categorical columns
    # print ("One-hot-encoding all categorical columns ...")
    # joined = pd.get_dummies(joined)
    print ("Shape of joined:", joined.shape)

    df_train = joined[:train.shape[0]]
    df_test = joined[train.shape[0]:]

    shift = 200
    y = np.log(df_train['loss'] + shift)
    ids = df_test['id']
    X = df_train.drop(['loss', 'id'], 1)
    X_test = df_test.drop(['loss', 'id'], 1)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=7)
    eval_set = [(x_val, y_val)]

    param_grid = dict(
        #n_estimators=[750, 1000],
        # max_depth=[6, 9],
        # learning_rate=[0.01, 0.001],
        # min_child_weight=[1,3,5]
    )

    print ("Starting to fit the model ... ")
    start = time.time()    
    xgbr = XGBRegressor(
        n_estimators=750,
        # learning_rate=0.01,
        max_depth=3,
        # min_child_weight=5
    )
    cv = KFold(n_splits=3, shuffle=True)
    clf = GridSearchCV(
        xgbr,
        param_grid,
        scoring='neg_mean_absolute_error',
        cv=cv,
        verbose=2,
        n_jobs=-1
    )
    '''
    xgbr.fit(
        x_train, 
        y_train,
        #eval_metric=['mae'],
        eval_set=eval_set,
        early_stopping_rounds=20,
        verbose=True)
    '''
    grid_result = clf.fit(x_train, y_train)
    end = time.time()
    print ("Time to fit the model:",time.strftime("%H:%M:%S", time.gmtime(end-start)))


    # y_pred = np.exp(xgbr.predict(x_val)) - shift
    # y_val_exp = np.exp(y_val) - shift
    # print ("Mean Absolute Error:", mean_absolute_error(y_val_exp, y_pred))

    y_pred = np.exp(clf.predict(x_val)) - shift
    y_val_exp = np.exp(y_val) - shift
    print ("Mean Absolute Error:", mean_absolute_error(y_val_exp, y_pred))

    print ("\nBest Estimator:", clf.best_estimator_)

    print ("\n\n", grid_result.cv_results_)


	
