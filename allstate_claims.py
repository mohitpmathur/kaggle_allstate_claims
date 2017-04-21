
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
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

    # for column in list(train.select_dtypes(include=['object']).columns):
    #     if set(train[column].unique()) != set(test[column].unique()):
    #         #if train[column].nunique() != test[column].nunique():
    #         print ("\n",column)
    #         set_train = set(train[column].unique())
    #         set_test = set(test[column].unique())
    #         remove_train = set_train - set_test
    #         remove_test = set_test - set_train
    #         remove = remove_train.union(remove_test)
    #         print (remove_train)
    #         print (remove_test)
    #         print (remove)
    #         def filter_cat(x):
    #             # Remove if exist in only either train or test
    #             if x in remove:
    #                 return np.nan
    #             return x
            
    #         joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)
    #         #print (joined[column].unique())
    #     #joined[column] = pd.factorize(joined[column].values, sort=True)[0]
    
    # One-hot-encode categorical columns
    print ("One-hot-encoding all categorical columns ...")
    joined = pd.get_dummies(joined)
    print ("Shape of joined:", joined.shape)

    df_train = joined[:train.shape[0]]
    df_test = joined[train.shape[0]:]

    shift = 200
    y = np.log(df_train['loss'] + shift)
    ids = df_test['id']
    X = df_train.drop(['loss', 'id'], 1)
    X_test = df_test.drop(['loss', 'id'], 1)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=7)

    print ("Starting to fit the model ... ")
    start = time.time()
    gbr = GradientBoostingRegressor(
        n_estimators=500, 
        warm_start=True,
        verbose=1)
    gbr.fit(x_train, y_train)
    end = time.time()
    print ("Time to fit the model:",time.strftime("%H:%M:%S", time.gmtime(end-start)))
    y_pred = np.exp(gbr.predict(x_val)) - shift
    y_val_exp = np.exp(y_val) - shift
    print ("Mean Absolute Error:", mean_absolute_error(y_val_exp, y_pred))


	
