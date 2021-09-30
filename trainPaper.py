#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:55:44 2019

@author: dario
"""

import pandas as pd
import numpy as np
import time
import gc
        
from keras import layers
from keras import regularizers
from keras import optimizers
from keras import models
import keras

def build_model(input_shape):

    reg_const = 0.0001
    activation_func = 'relu'

    wav_input = layers.Input(shape=input_shape, name='wav_input')
    
    conv1 = layers.Conv2D(32, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(wav_input)
    conv1 = layers.Conv2D(64, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)

    conv1 = layers.Conv2D(64, (39, 5), strides=(39, 5),  activation=activation_func, padding = 'same', kernel_regularizer=regularizers.l2(reg_const))(conv1)

    
    conv1 = layers.Flatten()(conv1)
    conv1 = layers.Dropout(0.4)(conv1)    
    
    
    conv1 = layers.Dense(128, activation=activation_func)(conv1)

    
    meta_input = layers.Input(shape=(1,), name='meta_input')
    meta = layers.Dense(1)(meta_input)



    merged = layers.concatenate(inputs=[conv1, meta])
    merged = layers.Dense(128, activation=activation_func)(merged)


    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    final_model = models.Model(inputs=[wav_input, meta_input], outputs=[pga, pgv, sa03, sa10, sa30]) #pga, pgv, sa03, sa10, sa30
    
    rmsprop = optimizers.RMSprop(lr=0.0001)

    final_model.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])
    
    return final_model


def build_pretrained_model(input_shape, stat_dists_shape):
    
    reg_const = 0.0001
    activation_func = 'relu'
    model = models.load_model('centIT10sSTead.hdf5') 

    conv1Weights = model.layers[1].get_weights() 
    conv2Weights = model.layers[2].get_weights() 
    
    wav_input = layers.Input(shape=input_shape, name='wav_input')


    conv1 = layers.Conv2D(32, (1, 125), strides=(1, 2), trainable=False, weights=conv1Weights, activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(wav_input)

    conv1 = layers.Conv2D(64, (1, 125), strides=(1, 2), trainable=False, weights=conv2Weights, activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)


    conv1 = layers.Conv2D(64, (39, 5), strides=(39, 5), activation=activation_func, padding = 'same', kernel_regularizer=regularizers.l2(reg_const))(conv1)

    conv1 = layers.Flatten()(conv1)
    
    conv1 = layers.Dropout(0.4)(conv1)    
    
    meta_input = layers.Input(shape=(1,), name='meta_input')
    meta = layers.Dense(1)(meta_input)

    stat_input = layers.Input(shape=stat_dists_shape, name='stat_input')

    merged = layers.concatenate(inputs=[conv1, meta, stat_input])
    merged = layers.Dense(128, activation=activation_func)(merged)

    
    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    #For training the single-station target model
    # pga = layers.Dense(1)(merged)
    # pgv = layers.Dense(1)(merged)
    # sa03 = layers.Dense(1)(merged)
    # sa10 = layers.Dense(1)(merged)
    # sa30 = layers.Dense(1)(merged)
    
    
    final_model = models.Model(inputs=[wav_input, meta_input, stat_input], outputs=[pga, pgv, sa03, sa10, sa30])
    
    rmsprop = optimizers.RMSprop(lr=0.0001)

    final_model.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])
    return final_model
     
def populate_array(element, length):
    newArray = []
    for i in range(0, length):
        newArray.append(element)
    
    return np.array(newArray)


def normalize(inputs):
    # Normalize the inputs and return the input maximum
    maxes = []
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        maxes.append(maks)
        if maks != 0:
            normalized.append(eq/maks)
        else:
            normalized.append(eq)
    return np.array(normalized), np.array(maxes)            
            

def targets_to_list(targets, singleStaTarg):
    if singleStaTarg != True:
        targets2 = targets.transpose(2,0,1)
    
        targetList = []
        for i in range(0, len(targets2)):
            targetList.append(targets2[i,:,:])
    
    else:
        targets2 = targets.transpose(1,0)
        targetList = list(targets2)
        
    return targetList

    
def targets_to_log(targets):
    targets = np.log10(targets)
    
    targetMin = targets[targets != -np.inf].min()
    targets[targets == -np.inf] = targetMin-1
    
    return targets
        
def merge_splits(inputs, targets, meta, k, numFolds):
    #Merge the existing k subsets into training (k-1) and test (1) subsets
    
    if k != 0:
        z=0
        inputsTrain = inputs[z]
        targetsTrain = targets[z]
        metaTrain = meta[z]
    else:
        z=1
        inputsTrain = inputs[z]
        targetsTrain = targets[z]
        metaTrain = meta[z]

    for i in range(z+1, numFolds):
        if i != k:
            inputsTrain = np.concatenate((inputsTrain, inputs[i]))
            targetsTrain = np.concatenate((targetsTrain, targets[i]))
            metaTrain = np.concatenate((metaTrain, meta[i]))
    
    return inputsTrain, targetsTrain, metaTrain, inputs[k], targets[k], meta[k]

def results_to_file(outputs_array, name, single_stat):
    # Output the results into a .csv file
    
    if single_stat != True:
        outputs_array = outputs_array.reshape((len(outputs_array[:,0,0])*len(outputs_array[0,:,0]), len(outputs_array[0,0,:])))
    
    outputsDF = pd.DataFrame(outputs_array, columns=[
        'pga', 'pgv', 'psa03', 'psa10', 'psa30', 'pred_pga', 'pred_pgv', 'pred_psa03', 'pred_psa10',
       'pred_psa30', 'event_id', 'sta', 'Observed',
       'Epicentral distance', 'Vs30', 'Magnitude', 'Depth', 'Latitude',
       'Longitude', 'pga_median', 'pgv_median','sa03_median','sa10_median','sa30_median','has_data', 'Filename'])

    outputsDF.drop(['pga_median', 'pgv_median','sa03_median','sa10_median','sa30_median','has_data', 'Filename'], inplace=True, axis=1)
    
    
    outputsDF.to_csv(name, index=False)
    
    return None

def construct_weight_array(trainMeta):
    # Make an array of sample weights based on the IM values to use during training
    eq_medians = []
    for eq in trainMeta:
        eq_medians.append(eq[0][9:14]*1000)
    
    eq_medians = np.array(eq_medians)
    eq_medians[:,1] = np.square(eq_medians[:,1])*10
    
    return eq_medians

def get_uniform_weights(nonWeighted):
    # Make an array of sample weights based on the weighted magnitudes to use during training

   values, weights = np.unique(nonWeighted, return_counts=True)
   weights = values*1/weights
     
   weightsDict = dict(zip(values, weights))
   
   weighted = []
   for val in nonWeighted:
       weighted.append(weightsDict[val])

   weightedReplicated = np.tile(weighted, (5, 1)).transpose()
   return np.array(weightedReplicated)

def add_mags(trainMeta, trainDists):
    # Adding perturbed magnitude as an additional input to the model
    trainMags = trainMeta[:,0,5]
    perturbations = np.random.uniform(-0.5, 0.5, len(trainMags))
    
    trainMags = trainMags+perturbations
    
    trainDists = np.append(trainDists, trainMags.reshape(-1,1).astype(np.float), axis=1)
    
    return trainDists

def main():
    # outputs = np.empty((0,21)) # Single station preds
    outputs = np.empty((0,39,26)) # Multi station with weights

    inputsK = []
    targetsK = []
    metaK = []

    numFolds = 5
    for i in range(numFolds):
        inputsK.append(np.load('npyData/' + str(i) + '/waveforms.npy'))
        targetsK.append(np.load('npyData/' + str(i) + '/targets.npy'))
        metaK.append(np.load('npyData/' + str(i) + '/meta.npy', allow_pickle=True))
    
    
    earlystop = keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=0.00001)
    
    
    stat_dists = pd.read_csv('stationDists.csv')
    stat_dists['sin_az'] =  np.sin(np.radians(stat_dists['Azimuth']))
    stat_dists['cos_az'] =  np.cos(np.radians(stat_dists['Azimuth']))
    stat_dists['Distance'] = stat_dists['Distance']/stat_dists['Distance'].max() # Dividing by maximum to have values around 1, with the idea of avoiding exploding gradients
     
    stat_dists = stat_dists[['Distance', 'sin_az', 'cos_az']]
    stat_dists = np.array(stat_dists).flatten()
    
    
    for k in range(0, numFolds):
        ###
        ## Split the data into training (by combining numFolds-1 subsets) and
        ## test set (1 subset)
        ###
        trainInputsAll, trainTargets, trainMeta, testInputsAll, testTargets, testMeta = merge_splits(inputsK, targetsK, metaK, k, numFolds)
         
        ###
        ## Make an array of interstation distances and azimuths to be inserted
        ## as inputs to the model
        ###
        trainDists = populate_array(np.array(stat_dists), len(trainInputsAll))
        testDists = populate_array(np.array(stat_dists), len(testInputsAll))
    
        ###
        ## Normalize the input data and get the maximum to use as additional input
        ###
        trainInputs, trainMaxes = normalize(trainInputsAll[:, :, :, :])
        testInputs, testMaxes = normalize(testInputsAll[:, :, :, :])
    
        trainMaxes = targets_to_log(trainMaxes)
        testMaxes = targets_to_log(testMaxes)
        
        ###
        ## Get sample weights for training the model
        ## to include them in the model training add the following argument
        ## to model.fit(...):
        ##   sample_weight={'pga':train_weights[:,0], 'pgv':train_weights[:,1], 'sa03':train_weights[:,2], 'sa10':train_weights[:,3], 'sa30':train_weights[:,4]},

        ## Weigths based on maximum of the IM
        #train_weights = construct_weight_array(trainMeta) 
        
        ## Weigths based on the weighted magnitudes
        #train_weights = get_uniform_weights(trainMeta[:,0,5])


    
        start_time = time.time()

        # trainTargets = trainTargets[:, 31, :] ## single station (PII) predict
        # testTargets = testTargets[:, 31, :] ## single station (PII) predict

        trainTargets = trainTargets[:, :, :] ## multi station predict
        testTargets = testTargets[:, :, :] ## multi station predict

        ###
        ## Train the model from scratch, i.e. the weights are randomly initialized
        ###
        # model = build_model(testInputs[0].shape)
        
        # history = model.fit(x=[trainInputs, trainMaxes], y=targets_to_list(trainTargets, False),
        #     epochs=1000, batch_size=8, verbose=0, callbacks=[earlystop, reduce_lr],
        #     validation_data=([testInputs, testMaxes], targets_to_list(testTargets, False)
        #                         ))

        # predictions = model.predict([testInputs, testMaxes])#, testDists])


        ###
        ## Train the model using Transfer Learning
        ###
        
        model = build_pretrained_model(testInputs[0].shape, trainDists[0].shape)

        history = model.fit(x=[trainInputs, trainMaxes, trainDists], y=targets_to_list(trainTargets, False),
              epochs=1000, batch_size=8, verbose=0, callbacks=[earlystop, reduce_lr], 
              validation_data=([testInputs, testMaxes, testDists], targets_to_list(testTargets, False)
                                  ))
       
        predictions = model.predict([testInputs, testMaxes, testDists])
        
        
        print('Fold number:' + str(k))
        print(history.history['loss'][-1])
        print(history.history['val_loss'][-1])
        print("--- %s seconds ---" % (time.time() - start_time))
        print(str(len(history.history['loss'])) + ' epochs')
        
        ## Multi station model
        predictions = np.transpose(np.array(predictions), (1,2,0))
        tempArr = np.concatenate((testTargets, predictions), axis=2)
        tempArr = np.concatenate((tempArr, testMeta),axis=2)
        
        ## Single station model (station PII)
        # predictions = np.transpose(np.array(predictions), (1,0,2))[:,:,0]
        # tempArr = np.concatenate((testTargets, predictions), axis=1)
        # testMeta = testMeta[:,31,:]
        # tempArr = np.concatenate((tempArr, testMeta),axis=1)
        
        
        outputs = np.append(outputs, tempArr, axis=0)

        
        del trainInputsAll, trainTargets, trainMeta, testInputsAll, testTargets, testMeta
        del trainInputs, testInputs
        
        keras.backend.clear_session()

        gc.collect()
        
        time.sleep(20)

        

    results_to_file(outputs, 'results.csv', False)



if __name__== "__main__" :
    main()