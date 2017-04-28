'''
This code uses dataset from the Kaggle 'Allstate Claims Severity' 
competition 
https://www.kaggle.com/c/allstate-claims-severity

This is an implementation of deep neural network to predict the loss.

Tuning them, I was able to get a MSA of 1135 on the validation set

'''

import time
import numpy as np
import pandas as pd
# import matplotlib as plt
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import metrics
from keras.wrappers.scikit_learn import KerasRegressor
import keras.backend as k
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

def custom_mae(y_true, y_pred):
    y_pred = k.exp(y_pred) - 200
    y_true = k.exp(y_true) - 200
    return k.mean(k.abs(y_true - y_pred))

def baseline():
    model = Sequential()
    model.add(Dense(130, input_dim=130, activation='relu'))
    model.add(Dense(65, activation='relu'))
    model.add(Dense(1))
    model.compile(
        loss='mean_absolute_error', 
        optimizer='rmsprop', 
        metrics=[custom_mae])

    # print (model.summary())
    return model

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
print ("Shape of joined:", joined.shape)

df_train = joined[:train.shape[0]]
df_test = joined[train.shape[0]:]

shift = 200
y = np.log(df_train['loss'] + shift)
ids = df_test['id']
X = df_train.drop(['loss', 'id'], 1)
X_test = df_test.drop(['loss', 'id'], 1)

x_train, x_val, y_train, y_val = train_test_split(
    X, 
    y, 
    test_size=.25, 
    random_state=7)
print ("Shape of x_train:", x_train.shape)
print ("Shape of y_train:", y_train.shape)
print ("Shape of x_val:", x_val.shape)
print ("Shape of y_val:", y_val.shape)

x_train = x_train.values
y_train = y_train.values
print ("Shape of x_train: {}; type(x_train): {}".format(
    x_train.shape, 
    type(x_train)))
print ("Shape of y_train: {}; type(y_train): {}".format(
    y_train.shape, 
    type(y_train)))

x_val = x_val.values
y_val = y_val.values
print ("Shape of x_val: {}; type(x_val): {}".format(
    x_val.shape, 
    type(x_val)))
print ("Shape of y_val: {}; type(y_val): {}".format(
    y_val.shape, 
    type(y_val)))

dimension = x_train.shape[1]


#model = baseline()
model = KerasRegressor(build_fn=baseline, epochs=2, batch_size=32, verbose=2)
# history = model.fit(x_train, 
#                     y_train, 
#                     #epochs=2, 
#                     #batch_size=32,
#                     #validation_data=(x_val, y_val),
#                     #verbose=2
#                     )
kfold = KFold(n_splits=3, random_state=42)
clf = GridSearchCV(estimator=model, param_grid={}, cv=kfold)
print ("Finished")
start = time.time()
results = clf.fit(x_train, y_train)
end = time.time()
print ("Time to fit the model:",time.strftime("%H:%M:%S", time.gmtime(end-start)))

y_pred = np.exp(clf.predict(x_val)) - shift
y_val_exp = np.exp(y_val) - shift
print ("Mean Absolute Error:", mean_absolute_error(y_val_exp, y_pred))

print ("\nBest Estimator:", clf.best_estimator_.get_params())
print ("\nResults:", results.cv_results_)
print ("Model Summary:", clf.best_estimator_.build_fn().summary())