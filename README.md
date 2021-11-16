# Nural-Network-for-structured-data
A machine learning toolkit dedicated to time-series data analysis in Python. The ANN model prformance is 
@author: aamani
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Python code for predicting the variability of the cloud cover in respone to the variability to the Oceanic Climatic Indecies  
import random 
import numpy as np 
import numpy as std
import matplotlib.pyplot as plt 
import pandas as pd
from pandas.plotting import autocorrelation_plot
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers.core import Dense 
from tensorflow.python.keras.models import Sequential
from scipy import signal
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import tensorflow
from numpy import sqrt
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

from  IPython import display
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt


import numpy as np

import pathlib
import shutil
import tempfile




# Load the saved dataframe containing all the input variables and target: 
#variables = ['Variable 1','Variable 2','Variable 3','Variable 4','Variable 5','Variable 6','Variable 7','Variable 8','sum']


#definition of the Oceanic climatic Indecies (variables). sum variable represents the cloud
#'Nino34', 'IOD','PDO','NPGO','ONI','SOI','Nino4','IPO','sum'

variables=['Nino34', 'IOD','PDO','NPGO','ONI','SOI','Nino4','IPO','sum']
df = pd.read_csv('C:/Indian_ElNino_2/yearly_average.csv', index_col=0)
df.columns = variables
df.index = pd.to_datetime(df.index)

# Plot the variables

plt.figure()
df.plot(ylim = (0,1),figsize=(8,4))
plt.ylabel('Value')
plt.legend(loc='center left', bbox_to_anchor=(1.0, .8))
plt.tight_layout()
plt.savefig('Plot_data.png')





# Calculate and plot correlation matrix
corrMatrix = df.corr()
plt.figure()
sns.heatmap(corrMatrix, annot=True,vmin=-1, vmax=1,cmap="coolwarm")
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Calculate the cross correlation between "target" and variable 1. Find location of maxima in cross correlation
# Should be approx 7 days, due to the 7 day time shift incorporated earlier
#corr = pd.DataFrame(signal.correlate((df['sum'] - df['sum'].mean())/(df['sum'].std()), (df['Variable3']-df['Variable3'].mean())/(df['Variable3'].std()), mode='full') / len(df))
#corr.index = corr.index - 128
#maxima = corr.idxmax(axis=0)
#
## Plot the calculated cross correlation, along with the identified peak in maximum cross correlation
#plt.figure()
#plt.plot(corr)
#plt.plot([maxima,maxima], [-1,1], 'r--', lw=2)
#plt.plot([0,0], [-1,1], 'k-', lw=2)
#plt.xlim([-100,100])
#plt.ylim([-0.1,1])
#plt.xlabel('Lag [Days]')
#plt.ylabel('Correlation')
#plt.title('Cross correlation: "Target" vs. "Variable 1"')
#plt.savefig('Cross_correlation.png')
#plt.show()


# Plot the autocorrelation of the target variable. 
plt.figure()
autocorrelation_plot(df['sum'])
plt.xlabel('Lag [Days]')
plt.ylim([-0.2,1])
plt.xlim([0,100])
plt.savefig('AutoCorrelationw.png')
plt.show()


# If you want to perform a Granger causality test, with either "target" or "variable" as the first value. 
# Read more on statsmodels.tsa.stattools for grangercausalitytest
#from statsmodels.tsa.stattools import grangercausalitytests
#test = grangercausalitytests(df[['Target', 'Variable 1']], maxlag=10)
#test = grangercausalitytests(df[['Variable 1', 'Target']], maxlag=10)



# Define target variable used by the machine learning model, and split the data into train and test set. 
target= 'sum'
test_split = 0.25

test_samp = np.int(np.floor(test_split*len(df)))
df_train = df[test_samp:]
df_test = df[:test_samp]

# Define "X" as input variables and "Y" as target variable
X_train = df_train.drop([target], axis = 1)
X_train= np.array(X_train)

X_test = df_test.drop([target], axis = 1)
X_test= np.array(X_test)

Y_train = pd.DataFrame(df_train[target])
Y_test = pd.DataFrame(df_test[target])




# Scale input variables using MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define ANN model:
# Input layer:
model=Sequential()
# First hidden layer, connected to input vector X.
model.add(Dense(64,activation='relu',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.001),
                input_shape=(X_train.shape[1],)
               )
         )
#layers.Dropout(0.5),

#five hidden layer
#model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform',  kernel_regularizer=regularizers.l2(0.001)))
#layers.Dropout(0.5),

model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
#model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
# output layer. 1 neuron for estimating a 1 dimensional continous variable output.
model.add(Dense(1,kernel_initializer='glorot_uniform',   kernel_regularizer=regularizers.l2(0.001)))

#no accuracy in regression model , only in classification
#logistic regression is a classification algorithm.
#Root mean square error: rmse = np.sqrt(mse)

model.compile(loss='mse',optimizer='adam',metrics=['mse','mae'])

#Earlystopping
earlystop_callback = EarlyStopping(
  monitor='val_loss', min_delta=0.0001,
  patience=10)

#shuffle train set


# Train model for 100 epochs, batch size of 100: 
NUM_EPOCHS=30 #(train setsize/batch size)
BATCH_SIZE=10

history=model.fit(np.array(X_train_scaled),np.array(Y_train),callbacks=[earlystop_callback],
                batch_size=BATCH_SIZE, 
                epochs=NUM_EPOCHS,
                validation_split=0.1,
                verbose=1
                )
print('Loss: ', history.history['loss'][-1], '\nVal_loss: ', history.history['val_loss'][-1],  '\nmae: ', history.history['mae'][-1])

# evaluate the model

# Plot training curve showing train/val loss during training 
plt.plot(history.history['loss'],'blue',label='Training loss')
plt.plot(history.history['val_loss'],'r',label='Test loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.savefig('Training_curve.png')
plt.show()

# Make predictions in the test set:
y_pred = pd.DataFrame(model.predict(X_test_scaled))
y_pred.index = Y_test.index

y_pred_Train=pd.DataFrame(model.predict(X_train_scaled))
y_pred_Train.index = Y_train.index
#RMSE

#from keras import backend as K
#Ypred=y_pred[0]
#ytest=Y_test['sum']
#def root_mean_squared_error(ytest, Ypred):
        #return K.sqrt(K.mean(K.square(Ypred - ytest)))
#ANNloss=root_mean_squared_error(ytest, Ypred)
#print("losss:",ANNloss)

# Plot real vs. predicted values
plt.figure()
plt.plot(Y_test, label='Real')
#plt.plot(Y_train)
plt.plot(y_pred, label='Predicted')
plt.plot([y_pred.head().index[0],y_pred.head().index[0]], [0,80], 'k--', lw=2, label='Test set')
plt.legend()
plt.ylim([0,1])
#plt.xlim([1988,1996])
plt.ylabel('Value')
plt.savefig('Real_vs_Predicted.png')
plt.show()


# Calculate percentage error betwen real and predicted values
perc_err = (( y_pred[0] - Y_test['sum']  ))
per_std=np.std(perc_err)
per_max=max(perc_err)
per_min=min(perc_err)

#Positive or negative
neg_count = len(list(filter(lambda x: (x < 0), perc_err)))
  
# we can also do len(list1) - neg_count
pos_count = len(list(filter(lambda x: (x >= 0), perc_err)))



# Plot figures comparing real and predicted values in scatterplot, as well as distribution of percentage error
f, (ax1,ax2) = plt.subplots(1,2, figsize=(8,6) )

ax1.scatter(y_pred,Y_test, c="r", alpha=0.5, marker='o', label='Predictions')
ax1.plot([Y_test.min(),Y_test.max()], [Y_test.min(),Y_test.max()], 'k--', lw=4, label='Real = Predicted')
ax1.set_ylabel('Real value')
ax1.set_xlabel('Predicted value')
ax1.set_title('Predicted vs. Real values')
ax1.set_ylim(0,0.25)
ax1.set_xlim(0,0.3) 
ax1.legend()

sns.distplot(perc_err, ax=ax2, color='r', fit=norm, kde=False, bins=15)
ax2.set_title(' Histogram over error')
ax2.set_xlabel('% deviation: Real - Predicted')
#ax2.set_xlim(-0.8,0.1)
#ax2.set_ylim(0,3.0)
plt.savefig('Real_vs_Pred_distribution.png')





# Define Random forest model:

# set the hyperparameters for the random forest model
params_RF = {'n_estimators':1000,
             'max_depth':20,
             'random_state':42,
             'verbose': 0}

# Train model 

model_RF = RandomForestRegressor(**params_RF)
model_RF.fit(X_train_scaled, Y_train)


# Make predictions on test set
ypred_rf = pd.DataFrame(model_RF.predict(X_test_scaled))
ypred_rf.index = Y_test.index

meanSquaredError=mean_squared_error(Y_test, ypred_rf )
print("MSE:", meanSquaredError)
length = 2018
xmarks = [i for i in range(1988,length+1,3)]
# Plot real vs. predicted values for both ANN and RF model
plt.figure()
plt.plot(Y_test, label='real')
plt.plot(Y_train)
plt.plot(y_pred, label='predicted ANN')
plt.plot(ypred_rf, label='predicted Random forest')
plt.plot(y_pred_Train, label='predicted') #new
plt.plot([y_pred.head().index[0],y_pred.head().index[0]], [0,80], 'k--', lw=2, label='Test set')
plt.ylim([0,1])
#plt.xticks(xmarks)
plt.legend()
plt.savefig('Real_vs_Predicted_TwoModels.png')
plt.show()

#predicted vs real

f, (ax1,ax2) = plt.subplots(1,2, figsize=(8,6) )

ax1.scatter(ypred_rf,Y_test, c="g", alpha=0.5, marker='o', label='Predictions')
ax1.plot([Y_test.min(),Y_test.max()], [Y_test.min(),Y_test.max()], 'k--', lw=4, label='Real = Predicted')
ax1.set_ylabel('Real value')
ax1.set_xlabel('Random Forest Predicted value')
ax1.set_title('RFPredicted vs. Real values')
ax1.set_ylim(0,0.25)
ax1.set_xlim(0,0.3) 
ax1.legend()

sns.distplot(perc_err, ax=ax2, color='r', fit=norm, kde=False, bins=15)
ax2.set_title(' Histogram over error')
ax2.set_xlabel('% deviation: Real - Predicted')
#ax2.set_xlim(-0.8,0.8)
plt.savefig('Real_vs_Pred_distribution.png')


# Let's calculate and visualize the feature importance for our random forest model: 
nrows = 1
ncols = 1
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(8,6));

names_regressors = [("Random Forest", model_RF)]

nregressors = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_regressors[nregressors][0]
        regressor = names_regressors[nregressors][1]
        indices = np.argsort(regressor.feature_importances_)[::-1][:40]
        fig = sns.barplot(y=df_train.columns[indices][:40],
                        x=regressor.feature_importances_[indices][:40] , 
                        orient='h',ax=axes);
        fig.set_xlabel("Relative importance",fontsize=12);
        fig.set_ylabel("Features",fontsize=12);
        fig.tick_params(labelsize=12);
        fig.set_title(name + " feature importance");
        nregressors += 1

plt.savefig('Feature_Importance.png')
