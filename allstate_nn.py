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

model = Sequential()
model.add(Dense(130, input_dim=dimension, activation='relu'))
model.add(Dense(130, activation='relu'))
model.add(Dense(1))
model.compile(
    loss='mean_absolute_error', 
    optimizer='adam', 
    metrics=[metrics.mae])

print (model.summary())

history = model.fit(x_train, 
                    y_train, 
                    epochs=2, 
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    verbose=1)

y_pred = np.exp(model.predict(x_val)) - shift
y_val_exp = np.exp(y_val) - shift
print ("Mean Absolute Error:", mean_absolute_error(y_val_exp, y_pred))