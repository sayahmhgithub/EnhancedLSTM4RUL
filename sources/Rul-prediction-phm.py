#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:51:25 2020

@author: ficos
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras,os
from keras.models import load_model,Sequential
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation
from keras.layers import Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

#%%
# the declaration Variables that will be used in uot work :
cols = ['unit', 'cycles', 'op_setting1', 'op_setting2', 'op_setting3', 's1', 's2', 's3', 's4', 's5', 's6'
        , 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']


features_cols=['cycles','op_setting1', 'op_setting2', 'op_setting3', 's1', 's2', 's3', 's4', 's5', 's6'
        , 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
file=1
#%%
# Functions :=================================================================
# function to reshape features into (samples, time steps, features) 
def generate_features(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, (num_elements-seq_length)+1), range(seq_length, num_elements+1)):
        yield data_matrix[start:stop, :]
# function to generate labels
def generate_target(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length-1:num_elements, :]

def compute_score(d):
    return np.sum(np.exp(d[d >= 0] / 10) - 1) + np.sum(np.exp((-1) * d[d < 0] / 13) - 1)
#%%
# Read the training and test an real rul datasets  : ==========================
if file==1:
    look_back=31
if file==2:
    look_back=21
if file==3:
    look_back=38
if file==4:
    look_back=19
train = pd.read_csv('/home/ficos/Bureau/Zerhouni. Noureddine/master2019-Guebli/CMAPSSData/train_FD00' + str(file) + '.txt', sep=" ", header=None)
test = pd.read_csv('/home/ficos/Bureau/Zerhouni. Noureddine/master2019-Guebli/CMAPSSData/test_FD00' + str(file) + '.txt', sep=" ", header=None)
rul= pd.read_csv('/home/ficos/Bureau/Zerhouni. Noureddine/master2019-Guebli/CMAPSSData/RUL_FD00' + str(file) + '.txt', sep=" ", header=None)
#drop the columns that consisted of missing values: ===========================
train.drop(train.columns[[-1,-2]], axis=1, inplace=True)
test.drop(test.columns[[-1,-2]], axis=1, inplace=True)
rul.drop(rul.columns[[1]], axis=1, inplace=True)
# And then we will give names to all columns: =================================
train.columns = cols
test.columns = cols
rul.columns=['rul']
#The sensors s1, s5, s10, s16, s18, and s19 as well as op_setting 3,
#, will  be removed bc they have no effects  (min=max)


# The distribution of the columns : ===========================================
# Establishing remaining life in cycles ======================================
train = pd.merge(train, train.groupby('unit', as_index=False)['cycles'].max(), how='left', on='unit')
train.rename(columns={"cycles_x": "cycles", "cycles_y": "maxcycles"}, inplace=True)
test = pd.merge(test, test.groupby('unit', as_index=False)['cycles'].max(), how='left', on='unit')
test.rename(columns={"cycles_x": "cycles", "cycles_y": "maxcycles"}, inplace=True)
#determining the time to failure (TTF) for every row
train['ttf2'] = train['maxcycles'] - train['cycles']

train['ttf'] = np.where(train['ttf2'] >= 130, 130, train.ttf2 )
train['ttf_per'] =train['ttf']/130

#%%
# copy the train and test dataset =============================================
train_copy = train.copy()
test_copy = test.copy()

#%% Scaling the train and test dataset using MinMaxScale ======================
scaler = MinMaxScaler()
train_copy.iloc[:,1:26]= scaler.fit_transform(train_copy.iloc[:,1:26]).round(3)
test_copy.iloc[:,1:26] = scaler.transform(test_copy.iloc[:,1:26]).round(3)



#%% Visualize the scaled and unscaled data ===================================

fig = plt.figure(figsize = (8, 5))
fig.add_subplot(1,2,1)
plt.plot(train[train.unit==11].s20,c='r')
#plt.plot(test[test.unit==1].s2,)

plt.legend(['Train','Test'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.ylabel('Original unit')
fig.add_subplot(1,2,2)
plt.plot(train_copy[train_copy.unit==11].s20)
#plt.plot(test_copy[test_copy.unit==1].s2)
plt.legend(['Scaled Train','Scaled Test'], bbox_to_anchor=(0., 1.02, 1., .102), 
           loc=3, mode="expand", borderaxespad=0)
plt.ylabel('Scaled unit')
plt.show()

#%% generator x_train:
seq_gen = (list(generate_features(train_copy[train_copy['unit']==id], look_back, features_cols)) 
           for id in train_copy['unit'].unique())
x_train = np.concatenate(list(seq_gen)).astype(np.float32)
#
## generator y_train:
label_gen = [generate_target(train_copy[train_copy['unit']==id], look_back, ['ttf_per']) 
             for id in train_copy['unit'].unique()]
y_train = np.concatenate(label_gen).astype(np.float32)
#

# generator x_test:

seq_gen_test = (list(generate_features(test_copy[test_copy['unit']==id], look_back, features_cols)) 
           for id in test_copy['unit'].unique())

x_test = np.concatenate(list(seq_gen_test)).astype(np.float32)


#%% # Neural network LSTM with Keras
model = Sequential()
model.add(LSTM(input_shape=(look_back, x_train.shape[2]), units=12, return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(units=10,return_sequences=True))
model.add(LSTM(units=7,return_sequences=True))
model.add(LSTM(units=2,return_sequences=False))
model.add(Dense(units=2))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')               
print(model.summary())

#%% feed and train the proposed model :
#early_stopping =keras.callbacks.EarlyStopping(monitor=<'val_loss', verbose=1,baseline=0.0001, mode='auto', restore_best_weights=True)
#history = model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=1,validation_split=0.01,callbacks=[early_stopping])
history = model.fit(x_train, y_train, epochs =150,batch_size=100, verbose=1)

#%% plot the model loss :   
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.legend(['Loss ',], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
plt.ylabel('cost values')

plt.xlabel('epochs')
    
#%%
#Predict the test ttf persontages 
print("\n*************************************************************************************************")
ttf_test_per = modelm.predict(x_test,verbose=1)

model.evaluate(x_train, y_train)
#%%  generator test_truncated datasets:
list_temp=[]
for id in range(1,len(test.unit.unique())+1):
    list_temp.append(remove_first_seq(test[test['unit']==id],look_back))
test_truncated = pd.DataFrame(np.concatenate(list(list_temp)))
# add the ttf list in the test_truncated sets as a column :
test_truncated.columns=['unit', 'cycles', 'op_setting1', 'op_setting2', 'op_setting3', 's1', 's2', 's3', 's4', 's5', 's6'
        , 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21','maxcycles']


test_truncated['ttf_test_per'] = ttf_test_per

test_truncated['maxpredcycles'] = totcycles(test_truncated)

test_truncated['RUL'] =test_truncated['ttf_test_per']*130
# generate a list of RUL predicted :

rul_pred = []
for i in range(1,len(test.unit.unique())+1):
    nrul_pred=test_truncated[test_truncated.unit==i].iloc[int(test_truncated[test_truncated.unit==i].cycles.max()-look_back),29]
    rul_pred.append(nrul_pred)
   #%% 
df1=pd.DataFrame(rul.index+1)
df1.columns=['unit']
df1['rul']=rul.rul
df1['rul_pw']=rul.rul
df1['rul_pred']=rul_pred
df1.loc[df1['rul_pw'] >= 130,'rul_pw'] = 130
df1['difference']=df1.rul_pred-df1.rul_pw
#%%
d=df1.difference
mseL=mean_squared_error(rul_pred, df1.rul)
csL=compute_score(d)
msePW=mean_squared_error(rul_pred,  df1.rul_pw)
csPW=compute_score(d)  

print("\n************************************************************************")
print("RUl \t\t\t RMSE \t\t\t SCORE")
print("************************************************************************")
print('Line<ar RUL \t\t',np.sqrt(mseL).round(4),' \t\t ',csL.round())
print("************************************************************************")
print('Piece-wise RUL \t\t',np.sqrt(msePW).round(4),' \t\t ',csPW.round())
print("************************************************************************")


#%%  plot the diffrence
 plt.figure(figsize = (15, 7))
 plt.plot(df1.rul_pw,c='r')
 plt.plot(df1.rul_pred,c='black')
 plt.xlabel('# Unit', fontsize=16)
 plt.xticks(fontsize=16)
 plt.ylabel('RUL', fontsize=16)
 plt.yticks(fontsize=16)

 plt.legend(['True_Rul ','Pred_Rul'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0)
# plt.xticks(np.arange(0,55,2), fontsize=10)
 plt.yticks(np.arange(0,140,5),fontsize=10)
 plt.show()
#%%
ttf_train_per = model.predict(x_train,verbose=1)
mse=mean_squared_error(ttf_train_per,y_train)
print('\nScore:',compute_score(ttf_train_per-y_train))       
print(' MSE : ',mse.round(4),'\n RMSE : ',np.sqrt(mse).round(4))

