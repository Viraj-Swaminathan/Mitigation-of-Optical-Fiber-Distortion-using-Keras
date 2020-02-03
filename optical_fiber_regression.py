# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:35:48 2019

@author: Viraj
"""

import matplotlib
 
# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import scipy.io as spio
from copy import copy, deepcopy
from keras import optimizers
from keras.optimizers import SGD
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import random

random.seed(50)

mat = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm.mat', squeeze_me=True)

mat2 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm_2.mat', squeeze_me=True)

mat3 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm_3.mat', squeeze_me=True)

mat4 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm_4.mat', squeeze_me=True)

mat5 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm_5.mat', squeeze_me=True)

mat6 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm_6.mat', squeeze_me=True)

mat7 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm_7.mat', squeeze_me=True)

mat8 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm_8.mat', squeeze_me=True)

mat9 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm_9.mat', squeeze_me=True)

mat10 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7_-6dBm_10.mat', squeeze_me=True)

def Adjust_Input_Target_values(sig_rec_X,sig_in_X,Nsamp,Ntap,Ntr):
    N_side=(Ntap-1)/2  # no. of tap-weights on each side

    numberOfOutputs=Nsamp*1 #this is m

    numberOfInputs=Nsamp*Ntap #this is n
    
    #Training data and responses
    trainingPoints=np.zeros((numberOfInputs,int(Ntr/Ntap)))
    trainingResponses=np.zeros((numberOfOutputs,int(Ntr/Ntap)))
    index=0
    response_shift=int(Nsamp*N_side)
    for i in range (0,Ntr*Nsamp,Nsamp*Ntap):
        index=index+1
        trainingPoints[:,index-1]=sig_rec_X[0,i:i+(Nsamp*Ntap)].transpose()
        trainingResponses[:,index-1]=sig_in_X[0,i+response_shift:i+response_shift+Nsamp*1].transpose()
    return [trainingPoints,trainingResponses]
   

scale = 1
yy_in_2_scaled=np.reshape(mat['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled=np.reshape(mat['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled=np.reshape(mat['yy_2']*scale, (-1, 114688))# Input to Neural Network

No_samples_per_symbol_scaled = mat['No_samples_per_symbol']

No_taps_scaled = mat['No_taps']

No_actual_training_sequence_scaled = mat['No_actual_training_sequence']
                                   

[Input1,Target1]=Adjust_Input_Target_values(yy_2_after_LBF_scaled,yy_in_2_scaled,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

yy_in_2_scaled_2=np.reshape(mat2['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_2=np.reshape(mat2['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_2 = np.reshape(mat2['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input2,Target2]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_2,yy_in_2_scaled_2,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

yy_in_2_scaled_3=np.reshape(mat3['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_3=np.reshape(mat3['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_3 = np.reshape(mat3['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input3,Target3]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_3,yy_in_2_scaled_3,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

yy_in_2_scaled_4=np.reshape(mat4['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_4=np.reshape(mat4['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_4 = np.reshape(mat4['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input4,Target4]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_4,yy_in_2_scaled_4,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

yy_in_2_scaled_5=np.reshape(mat5['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_5=np.reshape(mat5['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_5 = np.reshape(mat5['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input5,Target5]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_5,yy_in_2_scaled_5,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

yy_in_2_scaled_6=np.reshape(mat6['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_6=np.reshape(mat6['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_6 = np.reshape(mat6['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input6,Target6]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_6,yy_in_2_scaled_6,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

yy_in_2_scaled_7=np.reshape(mat7['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_7=np.reshape(mat7['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_7 = np.reshape(mat7['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input7,Target7]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_7,yy_in_2_scaled_7,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

yy_in_2_scaled_8=np.reshape(mat8['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_8=np.reshape(mat8['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_8 = np.reshape(mat8['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input8,Target8]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_8,yy_in_2_scaled_8,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

yy_in_2_scaled_9=np.reshape(mat9['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_9=np.reshape(mat9['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_9 = np.reshape(mat9['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input9,Target9]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_9,yy_in_2_scaled_9,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

yy_in_2_scaled_10=np.reshape(mat10['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_10=np.reshape(mat10['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_10 = np.reshape(mat10['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input10,Target10]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_10,yy_in_2_scaled_10,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

Input = np.concatenate((Input1, Input2, Input3, Input4, Input5, Input6, Input7, Input8, Input9, Input10), 1)

mat11 = spio.loadmat('training_data_NRZ_RC_0_6_Ns_8_no_Gaurd_forced_1_L_140Km_Ntaps_7.mat', squeeze_me=True)

yy_in_2_scaled_11=np.reshape(mat11['yy_in_2']*scale, (-1, 114688)) # Target of Neural Network

# B2B fitered
yy_B2B_filtered_2_scaled_11=np.reshape(mat11['yy_BBP_filtered_2']*scale, (-1, 114688)) # Target of Neural Network

yy_2_after_LBF_scaled_11 = np.reshape(mat11['yy_2']*scale, (-1, 114688))# Input to Neural Network

[Input11,Target11]=Adjust_Input_Target_values(yy_2_after_LBF_scaled_11,yy_in_2_scaled_11,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_training_sequence_scaled)

Target = np.concatenate((Target11, Target11, Target11, Target11, Target11, Target11, Target11, Target11, Target11, Target11),1)

x=np.transpose(Input)

y=np.transpose(Target)

# create scaler
scaler = StandardScaler()

# fit scaler on training dataset
scaler.fit(y)

# transform training dataset
y = scaler.transform(y)

##create model
model4 = Sequential()

##add model layer
model4.add(Dense(6, input_dim=14, kernel_initializer='normal', activation='relu'))
model4.add(Dense(2))
model4.summary()
###
####compile model using mse as a measure of model performance
model4.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

class stopAtLossValue(Callback):

        def on_batch_end(self, batch, logs={}):
            THR = 0.02 #Assign THR with the value at which you want to stop training.
            if logs.get('loss') <= THR:
                 self.model.stop_training = True
                 
stopAtLossValueObject = stopAtLossValue()              
mc = ModelCheckpoint('best_model3.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model4.fit(x, y, epochs=150000, batch_size=256,  verbose=1, validation_split=0.125, callbacks=[mc,stopAtLossValueObject])
#
## load saved best model
model4.save('best_model3.h5')

model4 = load_model('best_model3.h5')

mat11 = spio.loadmat('Received_signal_2_samples_per_bit_-6dBm.mat', squeeze_me=True) # load testing input data
def Adjust_InputTest_values(sig_in_X,Nsamp,Ntap,Ntest):
    numberOfInputs = Nsamp*Ntap # number of Inputs is 14
    testingPoints = np.zeros((numberOfInputs,int((Ntest-Ntap)+1))) # create a matrix of zeros with dimensions of 14 by 131072
    
    index=0
    
    for i in range (0,(Ntest-Ntap)*Nsamp + 1,Nsamp): # increment by 2 each time to shift to the right by 1 bit
        index=index+1
        testingPoints[:,index-1]=sig_in_X[0,i:i+(Nsamp*Ntap)].transpose()    
        
    return testingPoints   

yy_rec_2_scaled = np.reshape(mat11['yy_rec_2_scaled'], (-1, 262156))

No_actual_testing_sequence_scaled = 131078

testInput = Adjust_InputTest_values(yy_rec_2_scaled,No_samples_per_symbol_scaled,No_taps_scaled,No_actual_testing_sequence_scaled)

# predict model
testInput = testInput.transpose()
test = model4.predict(testInput)

#inverse transform the scaled values to obtain the original values
test=scaler.inverse_transform(test)
print(test)

test1D = test.ravel()
test1D = np.concatenate([[0,0,0,0,0,0],test1D,[0,0,0,0,0,0]]) # concatenate 3 bits of zeros to the beginning and end as guarding bits
print(test1D)


spio.savemat('test1D_6nodes_-6dBm_1.mat', dict(test1D = test1D)) # save the array to load on Matlab

mat3 = spio.loadmat('The_equalized_samples_Mahmoud_6_dBm_sSamples_per_bit',squeeze_me=True)
test_Mahmoud = np.reshape(mat3['sig_ANN_direct'],(-1, 262156))
test_Mahmoud_1D = test_Mahmoud.ravel()
error = []
des_Output = deepcopy(test_Mahmoud_1D)
for row in range(262156):
    if (des_Output[row] == 0):
        des_Output[row] = 1e-15
    error.append(abs((des_Output[row]-test1D[row])/des_Output[row]))
#error = error.transpose()
#print(error)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

#-6dBm
