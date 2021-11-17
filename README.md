# Hello 👋
# ML for time-series
A machine learning toolkit dedicated to time-series data analysis in Python.
## Describtion
 Tenorflow is used to create ANN model. 

## Installation

```pip install tensorflow```

or ```conda``` 

``` conda create --name tensorflow python = 3.5```

``` activate tensorflow```

## Getting dataset ready 
Load the saved dataframe containing all the input variables and target. The time coloumn is the index with a shift =12Mo. The time formate is MM/DD/YEAR. The input variables and target saved as .csv file.
``` ruby
>>> variables = ['Variable 1','Variable 2','Variable 3','Variable 4','Variable 5','Variable 6','Variable 7','Variable 8','sum']
>>> df = pd.read_csv('...csv', index_col=0)
>>> df.columns = variables
>>> df.index = pd.to_datetime(df.index)
```


### plot the variables

```
>>> plt.figure()
>>> df.plot(ylim = (0,1),figsize=(8,4))
>>> plt.ylabel('Value')
>>> plt.legend(loc='center left', bbox_to_anchor=(1.0, .8))
>>> plt.tight_layout()
>>> plt.savefig('Plot_data.png')
```


## train and test set. 
``` ruby
>>>target= 'sum'
>>>test_split = 0.25

>>>test_samp = np.int(np.floor(test_split*len(df)))
>>>df_train = df[test_samp:]
>>>df_test = df[:test_samp]
```

### Define "X" as input variables and "Y" as target variable
```ruby 
>>> X_train = df_train.drop([target], axis = 1)
>>> X_train= np.array(X_train)

>>> X_test = df_test.drop([target], axis = 1)
>>> X_test= np.array(X_test)

>>> Y_train = pd.DataFrame(df_train[target])
>>> Y_test = pd.DataFrame(df_test[target])
```

## Data Processing and Transformation

### Scale input variables using MinMaxScaler
```ruby
>>> scaler = MinMaxScaler().fit(X_train)
>>> X_train_scaled = scaler.transform(X_train)
>>> X_test_scaled = scaler.transform(X_test)
```


## Define ANN model:
Input layer:
```
>>> model=Sequential()
# First hidden layer, connected to input vector X.
>>>model.add(Dense(64,activation='relu',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.001),
                input_shape=(X_train.shape[1],)
               )
         )
>>>layers.Dropout(0.5),

five hidden layer
>>>model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform',  kernel_regularizer=regularizers.l2(0.001)))
>>>layers.Dropout(0.5),

>>>model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.001)))
>>>model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
>>>model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
>>>model.add(Dense(1,kernel_initializer='glorot_uniform',   kernel_regularizer=regularizers.l2(0.001)))
```

### applying Earlystopping
```earlystop_callback = EarlyStopping(
  monitor='val_loss', min_delta=0.0001,
  patience=10)
```
shuffle train set


# Train model for 30  batch size of 100: 
NUM_EPOCHS= #(train setsize/batch size)
BATCH_SIZE=10

history=model.fit(np.array(X_train_scaled),np.array(Y_train),callbacks=[earlystop_callback],
                batch_size=BATCH_SIZE, 
                epochs=NUM_EPOCHS,
                validation_split=0.1,
                verbose=1
                )
print('Loss: ', history.history['loss'][-1], '\nVal_loss: ', history.history['val_loss'][-1],  '\nmae: ', history.history['mae'][-1])




 
 ## Comparing the ANN model with different ML models

