#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:55:44 2019

@author: dario
"""

import os
import pandas as pd
import numpy as np
import obspy
import xml.etree.ElementTree
import urllib.request
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import scipy
import json
import h5py
import time
import gc
        
from keras import layers
from keras import regularizers
from keras import optimizers
from keras import initializers
from keras import models
from keras import backend
import keras

def get_uniform_weights(nonWeighted):
   values, weights = np.unique(nonWeighted, return_counts=True)
   weights = values*1/weights
     
   weightsDict = dict(zip(values, weights))
   
   weighted = []
   for val in nonWeighted:
       weighted.append(weightsDict[val])

   return np.array(weighted)

def get_eqs():
    rootDir='./data2/SM4calc/' 
        
    folder_list = [ item for item in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, item)) ]
    
    return folder_list

def get_stations():
    stations = pd.read_pickle('Intro/chosenStations.pkl')

    return stations

    
#def extract_sm_data(earthquakes):
#    
#    for eq in earthquakes['#EventID']:
#        try:
#            path = 'data2/SM4calc/' + str(eq)
#            f = h5py.File(path + '/shake_result.hdf','r')
#            pgas = f['arrays']['imts']['GREATER_OF_TWO_HORIZONTAL']['PGA']
#            pgvs = f['arrays']['imts']['GREATER_OF_TWO_HORIZONTAL']['PGV']
#            sa03 = f['arrays']['imts']['GREATER_OF_TWO_HORIZONTAL']['SA(0.3)']
#            sa10 = f['arrays']['imts']['GREATER_OF_TWO_HORIZONTAL']['SA(1.0)']
#            sa30 = f['arrays']['imts']['GREATER_OF_TWO_HORIZONTAL']['SA(3.0)']
#            
#            table = pd.DataFrame([pgas['ids'], pgas['mean'], pgvs['mean'], sa03['mean'], sa10['mean'], sa30['mean']]).transpose()
#            table.columns = ['ID', 'pga', 'pgv', 'sa(0.3)', 'sa(1.0)', 'sa(3.0)']
#            table['ID'] = table['ID'].str.decode("utf-8")
#            
#            table['pga'] = table['pga'].apply(lambda x: np.exp(x)*9.8)
#            table['sa(0.3)'] = table['sa(0.3)'].apply(lambda x: np.exp(x)*9.8)
#            table['sa(1.0)'] = table['sa(1.0)'].apply(lambda x: np.exp(x)*9.8)
#            table['sa(3.0)'] = table['sa(3.0)'].apply(lambda x: np.exp(x)*9.8)
#            
#            table['pgv'] = table['pgv'].apply(lambda x: np.exp(x)/100)
#            table.to_pickle(path + '/predictions.pkl')
#    
#    
#            uncert_table = pd.DataFrame([pgas['ids'], pgas['std'], pgvs['std'], sa03['std'], sa10['std'], sa30['std']]).transpose()
#            uncert_table.columns = ['ID', 'pga', 'pgv', 'sa(0.3)', 'sa(1.0)', 'sa(3.0)']
#            uncert_table['ID'] = uncert_table['ID'].str.decode("utf-8")
#            uncert_table.to_pickle(path + '/uncertainties.pkl')
#    
#            
#            stat_info = pd.DataFrame()
#            stat_info['Epicentral distance'] = list(f['arrays']['distances']['repi'])
#            stat_info['Vs30'] = list(f['arrays']['vs30'])
#            stat_info.to_pickle(path + '/station_info.pkl')
#            
#            info = json.loads(f['dictionaries']['info.json'].value)
#            eq_info = pd.DataFrame(index=range(39), columns=['Magnitude', 'Depth', 'Latitude', 'Longitude'])
#            eq_info['Magnitude'] = np.float(info['input']['event_information']['magnitude'])
#            eq_info['Depth'] = np.float(info['input']['event_information']['depth'])
#            eq_info['Latitude'] = np.float(info['input']['event_information']['latitude'])
#            eq_info['Longitude'] = np.float(info['input']['event_information']['longitude'])
#    
#            eq_info.to_pickle(path + '/eq_info.pkl')
#        
#        except Exception as e:
#            print('Error for event ' + str(eq) + ':')
#            print(e.__class__.__name__)
#            print(e) 


def filter_trace(trace):
    # trace.detrend("linear")
    # trace.taper(max_percentage=0.05, type="hann")
    # trace.filter('bandpass', freqmin=0.05, freqmax=10, corners=2, zerophase=True)    
    #trace.resample(100)
    

    return trace

def find_horizontal_max(channels, stationName):
    values = []
    for channel in channels:
        if channel['name'][-1] != 'Z':
            valuesRow = {}
            for pgm in channel['amplitudes']:
                valuesRow[pgm['name']] = pgm['value']
            values.append(valuesRow)
    
    values = pd.DataFrame(values)
    
    return [stationName, 0.098*values['pga'].max(), values['pgv'].max()/100, 0.098*values['sa(0.3)'].max(), 0.098*values['sa(1.0)'].max(), 0.098*values['sa(3.0)'].max()]

def fill_row(station, eq_id):
    values = {}
    
    preds = station['properties']['predictions']
    
    for pgm in preds:
        if pgm['name'] != 'mmi':
            if pgm['name'] != 'pgv':
                values['pred_' + pgm['name']] = 0.098*pgm['value']
            else:
                values['pred_' + pgm['name']] = 0.01*pgm['value']

    values = pd.DataFrame([values], columns = values.keys())
    
    observed = find_horizontal_max(station['properties']['channels'], station['properties']['code'])
    observed = pd.DataFrame([observed], columns=['sta', 'pga', 'pgv', 'psa03', 'psa10', 'psa30'])

    observed.insert(0, column='event_id', value=eq_id)

    ren_cols = {'pred_sa(0.3)':'pred_psa03', 'pred_sa(1.0)':'pred_psa10', 'pred_sa(3.0)':'pred_psa30'}
    values.rename(columns=ren_cols, inplace=True)
        
    final_row = pd.concat([observed, values], axis=1)
    
    return final_row

def other_preds(testMeta, name):
    columnsDF = ['event_id', 'sta', 'pga', 'pgv', 'psa03',
       'psa10', 'psa30', 'pred_pga', 'pred_psa30', 'pred_psa10', 'pred_pgv', 'pred_psa03']
    final_table = pd.DataFrame(columns = columnsDF)
       
    for row in testMeta:
        eq_id = row[0][0]
        
        with open('data2/SM4calc/' + eq_id + '/stationlist.json', 'r') as read_file:
            data = json.load(read_file)
        
        for station in row[:, 1]:
            for feature in data['features']:
                if feature['properties']['code'] == station:
                    try:
                        final_table = final_table.append(fill_row(feature, eq_id), ignore_index=True)
                    except:
                        pass
            
            
    for column in columnsDF[2:]:
        final_table[column] = np.log10(final_table[column])
        
    final_table.replace(-np.inf, -7, inplace=True)
    
    final_table.to_csv(name)
    
    return final_table
    
# def pga_from_shakemap(eq, stations):
    
#     with open('data2/SM4calc/' + eq + '/stationlist.json', 'r') as read_file:
#         data = json.load(read_file)
    
#     sm_path = 'data2/SM4calc/' + str(eq) 
#     sm_preds = pd.read_pickle(sm_path + '/predictions.pkl')
#     uncertainties = pd.read_pickle(sm_path + '/uncertainties.pkl')
#     stat_info = pd.read_pickle(sm_path + '/station_info.pkl')
#     eq_info = pd.read_pickle(sm_path + '/eq_info.pkl')
    
#     all_pgas = []
#     for station in data['features']:
#         try:
#             all_pgas.append(find_horizontal_max(station['properties']['channels'], station['properties']['code']))
#         except:
#             pass

#     all_pgas = pd.DataFrame(all_pgas, columns=['Station', 'PGA', 'PGV', 'SA(0.3)', 'SA(1.0)', 'SA(3.0)'])
    
#     pgms = []
#     meta = []
#     obs = []
#     for station in stations['sta']:
#         meta.append((eq, station))
#         if np.any(station in all_pgas['Station'].values) == True:
#             stationValues = all_pgas.loc[all_pgas['Station'] == station]
#             pgms.append(np.array(stationValues.iloc[:, 1:]))
#             obs.append(1)
#         else:
#             stationValues = sm_preds.loc[sm_preds['ID'] == station]
#             pgms.append(np.array(stationValues.iloc[:, 1:]))
#             obs.append(0)
            
#     meta = pd.DataFrame(meta, columns=['EventID', 'station'])
#     meta['Observed'] = obs
    
#     meta = pd.concat([meta, uncertainties, stat_info, eq_info], axis=1)
    
#     return np.array(pgms),meta

def pga_from_shakemap(eq):
    fileTog = pd.read_csv('data2/SM4calc/' + str(eq) + '/metaPGMsFinal.csv')    
    meta = fileTog.iloc[:,:9]
    meta['pga_median'] = np.max(fileTog['pga'])
    meta['pgv_median'] = np.max(fileTog['pgv'])
    meta['sa03'] = np.max(fileTog['sa03'])
    meta['sa10'] = np.max(fileTog['sa10'])
    meta['sa30'] = np.max(fileTog['sa30'])
    pgms = fileTog.iloc[:,9:].to_numpy()
    
    return pgms, meta
    
def construct_input(eq, stations, magnitude):
            
    wavLen = 2500 # In samples
     
    pga, meta = pga_from_shakemap(eq)#, stations)
    pga = np.reshape(pga, (39,5))
    has_data = []
    comb_input = []
    fileNames = []
     
    for i, row in stations.iterrows():
        
        if magnitude < 4:
            file_path = 'data2/HH/' + eq + '/' + row['net'] + '.' + row['sta'] + '.' + eq + '.mseed'
            file_path2 =  'data2/EH/' + eq + '/' + row['net'] + '.' + row['sta'] + '.' + eq + '.mseed'
            file_path3 =  'data2/HN/' + eq + '/' + row['net'] + '.' + row['sta'] + '.' + eq + '.mseed'
        else:
            file_path3 = 'data2/HN/' + eq + '/' + row['net'] + '.' + row['sta'] + '.' + eq + '.mseed'
            file_path2 =  'data2/EH/' + eq + '/' + row['net'] + '.' + row['sta'] + '.' + eq + '.mseed'
            file_path =  'data2/HH/' + eq + '/' + row['net'] + '.' + row['sta'] + '.' + eq + '.mseed'
        

        fileNames.append(file_path)

        if os.path.exists(file_path) == True:
            waveform = obspy.read(file_path)
     
            wav_np = []
            for trace in waveform:
                trace = filter_trace(trace)
                if trace.count() > wavLen:
                    wav_np.append(np.array(trace)[:wavLen])
                else:
                    wavFullLen = np.zeros(wavLen)
                    wavFullLen[0:trace.count()] = np.array(trace)
                    wav_np.append(wavFullLen)
            comb_input.append(np.array(wav_np))
            has_data.append(1)

        
        elif os.path.exists(file_path2) == True:
            waveform = obspy.read(file_path2)
    
     
            wav_np = []
            for trace in waveform:
                trace = filter_trace(trace)
                if trace.count() > wavLen:
                    wav_np.append(np.array(trace)[:wavLen])
                else:
                    wavFullLen = np.zeros(wavLen)
                    wavFullLen[0:trace.count()] = np.array(trace)
                    wav_np.append(wavFullLen)
                    
            comb_input.append(np.array(wav_np))
            has_data.append(1)

        elif os.path.exists(file_path3) == True:
            waveform = obspy.read(file_path3)
    
     
            wav_np = []
            for trace in waveform:
                trace = filter_trace(trace)
                if trace.count() > wavLen:
                    wav_np.append(np.array(trace)[:wavLen])
                else:
                    wavFullLen = np.zeros(wavLen)
                    wavFullLen[0:trace.count()] = np.array(trace)
                    wav_np.append(wavFullLen)
            comb_input.append(np.array(wav_np))
            has_data.append(1)

        else:   
            comb_input.append(np.zeros((3,wavLen)))
            has_data.append(0)
       
    meta['Has data'] = np.array(has_data)
    meta['FileName'] = np.array(fileNames)
    return np.array(comb_input), np.array(pga), meta

def get_data():
#    earthquakes = get_eqs()
    earthquakes = pd.read_csv('data2/eqs_Final.csv')
#    stations = get_stations()
    stations = pd.read_csv('data2/chosen.csv')

    # distant_eqs = pd.read_csv('Epi_dist_4_closest.csv')

    # distant_eqs = distant_eqs[distant_eqs['dist1'] > 5.5*13]
    
    Parrs = pd.read_csv('Parrs.csv')
    distant_eqs = Parrs[Parrs['Parr']>17]
    inputs = []
    targets = []
    meta = []
    
    for i, row in earthquakes.iterrows():
        
        if row['#EventID'] in list(distant_eqs['event_id'].astype(int)):
            pass
        else:
            waveforms, target, metaEq = construct_input(str(row['#EventID']), stations, row['Magnitude'])
            
            if np.max(metaEq['Has data']) > 0.5:
            
                meta.append(metaEq)
                
                inputs.append(waveforms)
        
                targets.append(target)
        
        
        
    return np.array(inputs).transpose(0,1,3,2), np.array(targets), meta


def build_model(input_shape):

    reg_const = 0.0001
    activation_func = 'relu'

#########    ### Convolutional network 1
    wav_input = layers.Input(shape=input_shape, name='wav_input')
    
    conv1 = layers.Conv2D(32, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(wav_input)
    # conv1 = layers.SpatialDropout2D(0.1)(conv1)
#    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(64, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)
    # conv1 = layers.SpatialDropout2D(0.4)(conv1)

#    conv1 = layers.BatchNormalization()(conv1)
#    conv1 = layers.Conv2D(64, (1, 15), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)

    conv1 = layers.Conv2D(64, (39, 5), strides=(39, 5),  activation=activation_func, padding = 'same', kernel_regularizer=regularizers.l2(reg_const))(conv1)
#    conv1 = layers.LeakyReLU(alpha=0.0001)(conv1)
#    conv1 = layers.BatchNormalization()(conv1)
    
    conv1 = layers.Flatten()(conv1)
    
    # conv1 = layers.Dropout(0.4)(conv1)
    # conv1 = layers.Dense(128, activation=activation_func)(conv1)

    
    meta_input = layers.Input(shape=(1,), name='meta_input')
    meta = layers.Dense(1)(meta_input)

#    stat_input = layers.Input(shape=(39,), name='stat_input')
#    stations = layers.Flatten()(stat_input)
    #stations = layers.Dense(128)(stat_input)

    merged = layers.concatenate(inputs=[conv1, meta])
    merged = layers.Dropout(0.4)(merged)
    merged = layers.Dense(128, activation=activation_func)(merged)


#    merged = layers.Dense(39)(merged)
#
#    final = layers.Dense(input_shape[0])(merged)
#    final_model = models.Model(inputs=[wav_input, meta_input], outputs=merged) #pga

    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    final_model = models.Model(inputs=[wav_input, meta_input], outputs=[pga, pgv, sa03, sa10, sa30]) #pga, pgv, sa03, sa10, sa30
    
    rmsprop = optimizers.RMSprop(lr=0.0001)

    final_model.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])
    
    return final_model

def build_pretrained_model(frozen_layers):
    model = models.load_model('centIT10sSTead.hdf5')
    
    if len(frozen_layers):
        for freeze_ind in frozen_layers:
             model.layers[freeze_ind].trainable = False
    
    rmsprop = optimizers.RMSprop(lr=0.0001)

    model.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])

    return model



def build_model2(input_shape):

    reg_const = 0.0001
    activation_func = 'relu'

#########    ### Convolutional network 1
    wav_input = layers.Input(shape=input_shape, name='wav_input')
    
    conv1 = layers.Conv2D(32, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(wav_input)
#    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(64, (1, 125), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)

#    conv1 = layers.BatchNormalization()(conv1)
#    conv1 = layers.Conv2D(64, (1, 15), strides=(1, 2),  activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)

    conv1 = layers.Conv2D(64, (39, 5), strides=(39, 5),  activation=activation_func, padding = 'same', kernel_regularizer=regularizers.l2(reg_const))(conv1)
#    conv1 = layers.LeakyReLU(alpha=0.0001)(conv1)
#    conv1 = layers.BatchNormalization()(conv1)
    
    conv1 = layers.Flatten()(conv1)
    conv1 = layers.Dropout(0.4)(conv1)    
    
    
    conv1 = layers.Dense(128, activation=activation_func)(conv1)

    
    meta_input = layers.Input(shape=(1,), name='meta_input')
    meta = layers.Dense(1)(meta_input)

#    stat_input = layers.Input(shape=(39,), name='stat_input')
#    stations = layers.Flatten()(stat_input)
    #stations = layers.Dense(128)(stat_input)

    merged = layers.concatenate(inputs=[conv1, meta])
    # merged = layers.Dropout(0.4)(merged)
    # merged = layers.Dense(128, activation=activation_func)(merged)


#    merged = layers.Dense(39)(merged)
#
#    final = layers.Dense(input_shape[0])(merged)
#    final_model = models.Model(inputs=[wav_input, meta_input], outputs=merged) #pga

    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    final_model = models.Model(inputs=[wav_input, meta_input], outputs=[pga, pgv, sa03, sa10, sa30]) #pga, pgv, sa03, sa10, sa30
    
    rmsprop = optimizers.RMSprop(lr=0.0001)

    final_model.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])
    
    return final_model


def build_pretrained_model2(input_shape, stat_dists_shape):
    
    reg_const = 0.0001
    activation_func = 'relu'
    model = models.load_model('10secMagCNNSTEADmin4.h5') #centIT10sSTead

    conv1Weights = model.layers[1].get_weights() 
    conv2Weights = model.layers[2].get_weights() 
    #conv3Weights = model.layers[3].get_weights() 
    
    wav_input = layers.Input(shape=input_shape, name='wav_input')


    conv1 = layers.Conv2D(32, (1, 125), strides=(1, 2), trainable=False, weights=conv1Weights, activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(wav_input)

    conv1 = layers.Conv2D(64, (1, 125), strides=(1, 2), trainable=True, weights=conv2Weights, activation=activation_func, kernel_regularizer=regularizers.l2(reg_const))(conv1)


    conv1 = layers.Conv2D(64, (39, 5), strides=(39, 5), activation=activation_func, padding = 'same', kernel_regularizer=regularizers.l2(reg_const))(conv1)

    conv1 = layers.Flatten()(conv1)
    
    # conv1 = layers.Dense(128, activation=activation_func)(conv1)
    conv1 = layers.Dropout(0.4)(conv1)    
    
    meta_input = layers.Input(shape=(1,), name='meta_input')
    meta = layers.Dense(1)(meta_input)

    stat_input = layers.Input(shape=stat_dists_shape, name='stat_input')

    merged = layers.concatenate(inputs=[conv1, meta, stat_input])
    # merged = layers.Dropout(0.4)(merged)
    merged = layers.Dense(128, activation=activation_func)(merged)

    
    pga = layers.Dense(39)(merged)
    pgv = layers.Dense(39)(merged)
    sa03 = layers.Dense(39)(merged)
    sa10 = layers.Dense(39)(merged)
    sa30 = layers.Dense(39)(merged)
    
    # pga = layers.Dense(1)(merged)
    # pgv = layers.Dense(1)(merged)
    # sa03 = layers.Dense(1)(merged)
    # sa10 = layers.Dense(1)(merged)
    # sa30 = layers.Dense(1)(merged)
    
    
    final_model = models.Model(inputs=[wav_input, meta_input, stat_input], outputs=[pga, pgv, sa03, sa10, sa30]) #pga, pgv, sa03, sa10, sa30
    
    rmsprop = optimizers.RMSprop(lr=0.0001)

    final_model.compile(optimizer=rmsprop, loss='mse', metrics=['accuracy'])
    return final_model
     
def populate_array(element, length):
    newArray = []
    for i in range(0, length):
        newArray.append(element)
    
    return np.array(newArray)


def normalize(inputs):
    maxes = []
    normalized = []
    for eq in inputs:
        maks = np.max(np.abs(eq))
        maxes.append(maks)
        if maks != 0:
            normalized.append(eq/maks)
        else:
            normalized.append(eq)
#    maxes = np.reshape(np.array(maxes), (len(maxes), 1))
    return np.array(normalized), np.array(maxes)            
            

def split_to_sets(inputs, targets, meta):
    meta = np.stack(meta)
    
    p = np.random.permutation(len(targets))
    
    inputs = inputs[p]
    targets = targets[p]
    meta = meta[p]
    
    trainSize = int(len(targets)*0.8)
    
    trainInputs = inputs[:trainSize, :, :, :]
    trainTargets = targets[:trainSize]
    trainMeta = meta[:trainSize]        
    testInputs = inputs[trainSize:, :, :, :]
    testTargets = targets[trainSize:]
    testMeta = meta[trainSize:] 
    
    return trainInputs, trainTargets, trainMeta, testInputs, testTargets, testMeta

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

def populate_table(meta, targets, predictions):
    columnsDF = ['event_id', 'sta', 'pga', 'pgv', 'psa03',
       'psa10', 'psa30', 'pred_pga', 'pred_pgv', 'pred_psa03',
       'pred_psa10', 'pred_psa30']
    final_table = pd.DataFrame(columns = columnsDF)
    
    predictions = np.array(predictions)
    predictions = predictions.transpose(1,2,0)
    
    for i in range(0, len(meta)):
        
        if len(targets[i].shape)==1:
            tarArray = np.reshape(targets[i], (39,1))
            predArray = np.reshape(predictions[i], (39,1))
        else:
            tarArray = targets[i]
            predArray = predictions[i]
        stacked = np.hstack((meta[i], tarArray, predArray))
        stacked = pd.DataFrame(stacked, columns = columnsDF)
        final_table = pd.concat([final_table, stacked], ignore_index=True)
        
    final_table.to_csv('res/log/multiPred/testResults.csv')
    return final_table

def populate_single_table(meta, targets, predictions, component):
    columnsDF = ['event_id', 'sta', component, 'pred_' + component]
    final_table = pd.DataFrame(columns = columnsDF)
    
    for i in range(0, len(meta)):
        if len(targets[i].shape)==1:
            tarArray = np.reshape(targets[i], (39,1))
            predArray = np.reshape(predictions[i], (39,1))
    
        stacked = np.hstack((meta[i], tarArray, predArray))
        stacked = pd.DataFrame(stacked, columns = columnsDF)
        final_table = pd.concat([final_table, stacked], ignore_index=True)
    
    final_table.to_csv('res/log/singlePred/' + component +'testResults.csv')

    
    return final_table
    
def targets_to_log(targets):
    targets = np.log10(targets)
    
    targetMin = targets[targets != -np.inf].min()
    targets[targets == -np.inf] = targetMin-1
    
    return targets

def k_fold_split(inputs, targets, meta, numFolds):
    meta = np.stack(meta)
    
    p = np.random.permutation(len(targets))
    
    inputs = inputs[p]
    targets = targets[p]
    meta = meta[p]
    
    ind = int(len(inputs)/numFolds)
    inputsK = []
    targetsK = []
    metaK = []
    for i in range(0,numFolds-1):
        inputsK.append(inputs[i*ind:(i+1)*ind])
        targetsK.append(targets[i*ind:(i+1)*ind])
        metaK.append(meta[i*ind:(i+1)*ind])
    
    inputsK.append(inputs[(i+1)*ind:])
    targetsK.append(targets[(i+1)*ind:])
    metaK.append(meta[(i+1)*ind:])    
    
    return inputsK, targetsK, metaK
        
def merge_splits(inputs, targets, meta, k, numFolds):
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

def data_to_file(meta, targets, predictions, name):
    predictions = np.array(predictions)
    predictions = np.transpose(predictions, (1,2,0))
    
    newMeta = np.concatenate((meta, targets), axis=2)
    newMeta = np.concatenate((newMeta, predictions), axis=2)
    
    newMeta = newMeta.reshape((len(newMeta[:,0,0])*len(newMeta[0,:,0]), len(newMeta[0,0,:])))
    
    newMeta = pd.DataFrame(newMeta, columns=['event_id', 'sta', 'Observed', 'station','pgaUNC',
       'pgvUNC', 'sa(0.3)UNC', 'sa(1.0)UNC', 'sa(3.0)UNC',
       'Epicentral distance', 'Vs30', 'Magnitude', 'Depth', 'Latitude',
       'Longitude', 'has_data', 'Filename', 'pga', 'pgv', 'psa03', 'psa10', 'psa30', 'pred_pga', 'pred_pgv', 'pred_psa03', 'pred_psa10',
       'pred_psa30'])

    newMeta.drop(['station'], inplace=True, axis=1)
    
    newMeta.to_csv(name)
    
    return newMeta

def results_to_file(outputs_array, name, single_stat):
    
    if single_stat != True:
        outputs_array = outputs_array.reshape((len(outputs_array[:,0,0])*len(outputs_array[0,:,0]), len(outputs_array[0,0,:])))
    
    outputsDF = pd.DataFrame(outputs_array, columns=[
        'pga', 'pgv', 'psa03', 'psa10', 'psa30', 'pred_pga', 'pred_pgv', 'pred_psa03', 'pred_psa10',
       'pred_psa30', 'event_id', 'sta', 'Observed',
       'Epicentral distance', 'Vs30', 'Magnitude', 'Depth', 'Latitude',
       'Longitude', 'pga_median', 'pgv_median','sa03_median','sa10_median','sa30_median','has_data', 'Filename'])

    # outputsDF.drop(['station'], inplace=True, axis=1)
    
    outputsDF.to_csv(name, index=False)
    
    return None

def construct_weight_array(trainMeta):
    eq_medians = []
    for eq in trainMeta:
        eq_medians.append(eq[0][9:14]*1000)
    
    eq_medians = np.array(eq_medians)
    eq_medians[:,1] = np.square(eq_medians[:,1])*10
    
    return eq_medians

def add_mags(trainMeta, trainDists):
    trainMags = trainMeta[:,0,5]
    perturbations = np.random.uniform(-0.5, 0.5, len(trainMags))
    
    trainMags = trainMags+perturbations
    
    trainDists = np.append(trainDists, trainMags.reshape(-1,1).astype(np.float), axis=1)
    
    return trainDists

def main():
    # Normal training
    inputs, targets, meta = get_data()
    
    Parrs = pd.read_csv('Parrs.csv')
    
    tempInput = []
    for i in range(len(meta)):
        eq = int(meta[i].iloc[0]['EventID'])
        arrTime = int(100*Parrs[Parrs['event_id'] ==eq ]['Parr'])
        if arrTime > 301:
            startTime = arrTime-300
        else:
            startTime = 0

        tempInput.append(inputs[i, :, startTime:startTime+1000, :])
    
    inputs = np.array(tempInput)
    del tempInput
    targets = targets_to_log(targets)


#    targets = targets.transpose(0, 2, 1)
    numFolds = 5

    inputsK, targetsK, metaK = k_fold_split(inputs, targets, meta, numFolds)
    del inputs

    
    #outputs = np.empty((0,39,21)) # Multi station preds
    # outputs = np.empty((0,21)) # Single station preds
    outputs = np.empty((0,39,26)) # Multi station with weights

    
    earlystop = keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, min_lr=0.00001)
    
    
    stat_dists = pd.read_csv('stationDists.csv')
    stat_dists['sin_az'] =  np.sin(np.radians(stat_dists['Azimuth']))
    stat_dists['cos_az'] =  np.cos(np.radians(stat_dists['Azimuth']))
    stat_dists['Distance'] = stat_dists['Distance']/stat_dists['Distance'].max() # Dividing by maximum to have values around 1, with the idea of avoiding exploding gradients
     
    stat_dists = stat_dists[['Distance', 'sin_az', 'cos_az']]
    stat_dists = np.array(stat_dists).flatten()
    
    
    inputsK = []
    targetsK = []
    metaK = []

    numFolds = 5
    for i in range(5):
        inputsK.append(np.load('npyData/' + str(i) + '/waveforms.npy'))
        targetsK.append(np.load('npyData/' + str(i) + '/targets.npy'))
        metaK.append(np.load('npyData/' + str(i) + '/meta.npy', allow_pickle=True))
    
    for k in range(0, numFolds):
        trainInputsAll, trainTargets, trainMeta, testInputsAll, testTargets, testMeta = merge_splits(inputsK, targetsK, metaK, k, numFolds)
    
#        stat_dists = pd.read_pickle('distTo1stStation.pkl')
            
        trainDists = populate_array(np.array(stat_dists), len(trainInputsAll))
        testDists = populate_array(np.array(stat_dists), len(testInputsAll))
    
        # trainDists = add_mags(trainMeta, trainDists)
        # testDists = add_mags(testMeta, testDists)
    
    
        trainInputs, trainMaxes = normalize(trainInputsAll[:, :, :, :])
        testInputs, testMaxes = normalize(testInputsAll[:, :, :, :])
    
        trainMaxes = targets_to_log(trainMaxes)
        testMaxes = targets_to_log(testMaxes)
        
        train_weights = construct_weight_array(trainMeta)
    
        # trainMags = trainMeta[:,0,5]
        # train_weights = get_uniform_weights(trainMags)

        
    
        start_time = time.time()

        # trainTargets = trainTargets[:, 31, :] #single station pred
        # testTargets = testTargets[:, 31, :] #single station pred


        trainTargets = trainTargets[:, :, :] #multi station pred
        testTargets = testTargets[:, :, :]

        model = build_model2(testInputs[0].shape)

        # model = build_pretrained_model([1,2])
        #model = build_pretrained_model2(testInputs[0].shape, trainDists[0].shape)

        history = model.fit(x=[trainInputs, trainMaxes], y=targets_to_list(trainTargets, False),
                          #y=trainTargets,
            epochs=1000, batch_size=8, verbose=0, callbacks=[earlystop, reduce_lr],
            validation_data=([testInputs, testMaxes], targets_to_list(testTargets, False)
                                #testTargets
                                ))#t
        # history = model.fit(x=[trainInputs, trainMaxes, trainDists], y=targets_to_list(trainTargets, False),
        #                     #y=trainTargets,
        #       epochs=1000, batch_size=8, verbose=0, callbacks=[earlystop, reduce_lr], 
        #       # sample_weight={'pga':train_weights, 'pgv':train_weights, 'sa03':train_weights, 'sa10':train_weights, 'sa30':train_weights},
        #       # sample_weight={'pga':train_weights[:,0], 'pgv':train_weights[:,1], 'sa03':train_weights[:,2], 'sa10':train_weights[:,3], 'sa30':train_weights[:,4]},
        #       validation_data=([testInputs, testMaxes, testDists], targets_to_list(testTargets, False)
        #                           #testTargets
        #                           ))#t
       
        
        # for layer in model.layers:
        #     layer.trainable = True
        # model.compile(optimizer=optimizers.RMSprop(lr=0.00004), loss='mse', metrics=['accuracy'])
        
        # history = model.fit(x=[trainInputs, trainMaxes, trainDists], y=targets_to_list(trainTargets),
        #                     #y=trainTargets,
        #       epochs=1000, batch_size=8, verbose=0, callbacks=[earlystop],
        #       validation_data=([testInputs, testMaxes, testDists], targets_to_list(testTargets)
        #                           #testTargets
        #                           ))
        
        print('Fold number:' + str(k))
        print(history.history['loss'][-1])
        print(history.history['val_loss'][-1])
        print("--- %s seconds ---" % (time.time() - start_time))
        print(str(len(history.history['loss'])) + ' epochs')
        
        # plt.plot(history.history['loss'][-30:])
        # plt.plot(history.history['val_loss'][-30:])
        # plt.title('model train vs validation loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper right')
        # plt.show()
        
        # plt.plot(history.history['loss'][10:])
        # plt.plot(history.history['val_loss'][10:])
        # plt.title('model train vs validation loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper right')
        # plt.show()
        
        # plt.close('all')
        ## Multi station model
        predictions = model.predict([testInputs, testMaxes])#, testDists])
        predictions = np.transpose(np.array(predictions), (1,2,0))
        tempArr = np.concatenate((testTargets, predictions), axis=2)
        tempArr = np.concatenate((tempArr, testMeta),axis=2)
        
        # Single station model
        # predictions = model.predict([testInputs, testMaxes, testDists])
        # predictions = np.transpose(np.array(predictions), (1,0,2))[:,:,0]
        # tempArr = np.concatenate((testTargets, predictions), axis=1)
        # testMeta = testMeta[:,31,:]
        # tempArr = np.concatenate((tempArr, testMeta),axis=1)
        
        
        
        outputs = np.append(outputs, tempArr, axis=0)

        
        time.sleep(20)
        
        del trainInputsAll, trainTargets, trainMeta, testInputsAll, testTargets, testMeta
        del trainInputs, testInputs
        
        keras.backend.clear_session()

        gc.collect()

        

        # fileName = 'res/SManalysis/15skFoldML' + str(k) + '.csv'
#        fileNameGMPE = 'res/SManalysis/15skFoldGMPE' + str(k) + '.csv'
        
#        data_to_file(testMeta, testTargets, predictions, fileName)

    results_to_file(outputs, 'Res/thesis/10sfScratchStrideOld.csv', False)
    # other_preds(np.stack(meta), 'Res/GMPEpredsChosen.csv')

#    specInputs = inputs[528:, :, :, :]
#    specTargets = targets[528:, :, :]
#    specMeta = meta[528:]
#    
#    chosInputs = inputs[:528, :, :, :]
#    chosTargets = targets[:528, :, :]
#    chosMeta = meta[:528]
#    
#    model = build_model(specInputs[0].shape)
#    
#    specInputsN, specMaxes = normalize(specInputs[:, :, :1000, :])
#    chosInputsN, chosMaxes = normalize(chosInputs[:, :1000, :])
#    
#    
#    history = model.fit(x=[specInputsN, specMaxes], y=targets_to_list(specTargets),
#                            #y=trainTargets,
#              epochs=30, batch_size=5,
#              validation_data=([chosInputsN, chosMaxes], targets_to_list(chosTargets)))
#
#    predictions = model.predict([chosInputsN, chosMaxes])


#    history = model.fit(x=[specInputsN, specMaxes], y=targets_to_list(specTargets),
#                            #y=trainTargets,
#              epochs=30, batch_size=5,
#              validation_data=([np.expand_dims(chosInputsN, axis=0), np.expand_dims(np.max(chosMaxes), axis=0)], targets_to_list(np.expand_dims(chosTargets, axis=0))))
#
#    predictions = model.predict([np.expand_dims(chosInputsN, axis=0), np.expand_dims(np.max(chosMaxes), axis=0)])

#   
#    plt.figure(figsize=(10,10))
#    plt.scatter(np.expand_dims(chosTargets, axis=0)[:,:,0], predictions[:,:,0], zorder=10, s=3)
##    plt.plot([1,1000000], [1,1000000], linestyle='-', linewidth=0.8, zorder=4)
#    plt.title('Predicted vs observed log(PGA)')
#    plt.ylabel('predicted')
#    plt.xlabel('observed')
##    plt.xlim(1, 1000000)   
##    plt.ylim(1, 1000000)
#    plt.grid(True, linestyle='-', linewidth=0.5)
#    plt.show()        

#     plt.figure(figsize=(10,10))
#     plt.scatter(outputs[:,:, 0].astype(float), outputs[:,:, 1].astype(float), c=outputs[:,:,2].astype(float), zorder=10, s=3)
# #    plt.plot([1,1000000], [1,1000000], linestyle='-', linewidth=0.8, zorder=4)
#     plt.title('Predicted vs observed log(PSA 3.0)')
#     plt.ylabel('predicted')
#     plt.xlabel('observed')
#     plt.xlim(-6.5, 1.5)
#     plt.ylim(-6.5, 1.5)
#     plt.grid(True, linestyle='-', linewidth=0.5)
#     plt.plot([-7, 1.5], [-7, 1.5], ls="--", c=".3")
#     plt.show() 
 
    
#     plt.figure(figsize=(10,10))
#     plt.scatter(testTargets[:,:], predictions[:,:], c=testMeta[:,:,0], zorder=10, s=3)
# #    plt.plot([1,1000000], [1,1000000], linestyle='-', linewidth=0.8, zorder=4)
#     plt.title('Predicted vs observed log(PSA 3.0)')
#     plt.ylabel('predicted')
#     plt.xlabel('observed')
#     plt.xlim(-6.5, 1.5)
#     plt.ylim(-6.5, 1.5)
#     plt.grid(True, linestyle='-', linewidth=0.5)
#     plt.plot([-7, 1.5], [-7, 1.5], ls="--", c=".3")
#     plt.show() 

#     plt.figure(figsize=(10,10))
#     plt.scatter(checkDF['Observed'], checkDF['Predicted'], zorder=10, s=3)
# #    plt.plot([1,1000000], [1,1000000], linestyle='-', linewidth=0.8, zorder=4)
#     plt.title('Predicted vs observed log(PGA)')
#     plt.ylabel('predicted')
#     plt.xlabel('observed')
# #    plt.xlim(1, 1000000)   
# #    plt.ylim(1, 1000000)
#     plt.grid(True, linestyle='-', linewidth=0.5)
#     plt.show()      

#     plt.plot(history.history['loss'][1:])
#     plt.plot(history.history['val_loss'][1:])
#     plt.title('model train vs validation loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper right')
#     plt.show()
# #    
#     plt.figure(figsize=(10,10))
#     plt.scatter(testTargets[:,:,1], predictions[1], zorder=10, s=3)
# #    plt.plot([1,1000000], [1,1000000], linestyle='-', linewidth=0.8, zorder=4)
#     plt.title('Predicted vs observed log(PGV)')
#     plt.ylabel('predicted')
#     plt.xlabel('observed')
# #    plt.xlim(1, 1000000)   
# #    plt.ylim(1, 1000000)
#     plt.grid(True, linestyle='-', linewidth=0.5)
#     plt.show()
    
#     plt.figure(figsize=(10,10))
#     plt.scatter(testTargets[:,:,0], np.array(predictions)[0,:,:], zorder=10, s=3)
# #    plt.plot([1,1000000], [1,1000000], linestyle='-', linewidth=0.8, zorder=4)
#     plt.title('Predicted vs observed log(PGA) - 10 sec', fontsize=26)
#     plt.ylabel('predicted', fontsize=20)
#     plt.xlabel('observed', fontsize=20)
#     plt.xlim(-6, 1)   
#     plt.ylim(-6, 1)
#     plt.grid(True, linestyle='-', linewidth=0.5)
#     plt.show()

# #    testTable = populate_table(testMeta, testTargets, predictions)
#     testTable = populate_single_table(testMeta, testTargets[:,:,0], predictions, 'sa30')


# import h5py
# f = h5py.File('10secMagCNNSTEADmin4.h5','r+')
# data_p = f.attrs['training_config']
# data_p = data_p.decode().replace("learning_rate","lr").encode()
# f.attrs['training_config'] = data_p
# f.close()

# startTime = 5.022585-3
# endTime7 = startTime + 10
# endTime4 = startTime + 7
# piiP = 11.446616
# piiPGA = 20.8403
# # def input_figure(seismos):
# #     for j in range(0,20):
# #        supt_text = 'Input example (M=3.3, Closest station at 79 km)' #str(meta[j].iloc[0,0])
# j = 94 # 4875411
# eq = 'NonNormInputExample' + str(meta[j].iloc[0,0])
# stations = list(meta[j]['station'])
# test = inputs[j]
# #test = test/np.max(np.abs(test))
# time_arr = np.arange(0,25,0.01)

# fig, axes = plt.subplots(39, 3, figsize=(15,10))

# i=0
# for row in axes:
#     for col in range(0,3):
#         row[col].plot(time_arr, test[i, :, col])
#         row[col].set_yticks([])
#         row[col].axvline(startTime, 0, 1, color='orange', label='Input waveform start time')
#         row[col].axvline(endTime4, 0, 1, color='green', label='Input waveform end time (7s)')
#         row[col].axvline(endTime7, 0, 1, color='#A971A8', label='Input waveform end time (10s)')
        
#         if stations[i] == 'PII':
#             row[col].axvline(piiP, 0, 1, color='blue', label='P arrival at PII')
#             row[col].axvline(piiPGA, 0, 1, color='red', label='PGA at PII')
#             handles, labels = row[col].get_legend_handles_labels()

#         #row[col].set_ylim([-1,1])
#         if i != 38:
#             row[col].set_xticks([])
#     h = row[0].set_ylabel(stations[i])
#     h.set_rotation(0)   
#     row[0].get_yaxis().set_label_coords(-0.1,0.0)
    
#     i += 1
    
# fig.suptitle('Non-normalised Input Example', fontsize=22)
# axes[0][1].set_title('M=4.4, Closest station ZCCA 27 km, PII 66 km', fontsize=15)
# axes[38][0].set_xlabel('Time (s)')
# axes[38][1].set_xlabel('Time (s)')
# axes[38][2].set_xlabel('Time (s)')
# fig.legend(handles, labels)#, loc='upper left')


# #fig.tight_layout() 
# fig.show()
# plt.savefig('Intro/' + eq + '.pdf')

# 4875411
# for j in range(266):
#     if meta[j].iloc[0,0] == 4875411:
#         print(j)


# for i in range(5):
#     np.save('npyData/' + str(i) + '/waveforms.npy', inputsK[i])
#     np.save('npyData/' + str(i) + '/targets.npy', targetsK[i])
#     np.save('npyData/' + str(i) + '/meta.npy', metaK[i])



if __name__== "__main__" :
    main()