##################
# Required Package
import numpy as np
import pandas as pd
import h5py
import json
import subprocess
import os.path
import cell_ontology as co
import time
import datetime
import os
import re
from scipy.optimize import curve_fit
import random
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pipeline2MyVersion as V6
import process_tsvs_v2_Normal as process_Normal
import process_tsvs_v2_Weight as process_Weight

##############################
# Train the STD-Mean Exp model
print("Start training std-mean model training.")
def get_signatures(index_list):
    """
    For a given list of indices in the file, returns a numpy array of
    gene signatures at those indices.
    """
    cpm = h5py.File(file_name, 'r')

    first = True
    for x in index_list:
        if first:
            signatures = cpm.get('cpm')[x]
            first = False
        else: 
            b = np.array(cpm.get('cpm')[x])
            signatures = np.column_stack([signatures, b])

    cpm.close()
    return signatures

# Provided h5 file
file_name = 'expression_CPM.h5'
decon_temp = './decon_temp/'
decon_temp_shell = "./decon_temp/"

# Load h5 data
cpm = h5py.File("expression_CPM.h5", 'r')
studies = np.array(cpm.get('study')).astype(str)
exp_acc = np.array(cpm.get('experiment_accession')).astype(str)
gene_ids = np.array(cpm.get('gene')).astype(str)
countspermillion = np.array(cpm.get('cpm'))
cpm.close()

with open('cell_types.json', 'r') as type_file:
    cell_type_file = json.load(type_file)
    
# Eliminate the redundant cell type in all exp
cell_type_specific_file = {}
for i in cell_type_file:
    cell_type_specific_file[i] = co.get_terms_without_children(cell_type_file[i])

cellExpDict = {}
for i in cell_type_specific_file:
    if cell_type_specific_file[i] == ['CL:2000001']:
        cellExpDict[i] = ['CL:2000001']

# Build the exp to study check dictionary
studyexpMap = {}
expstudyMap = {}
for i in range(len(exp_acc)):
    expstudyMap[exp_acc[i]] = studies[i]
    if studies[i] not in studyexpMap:
        studyexpMap[studies[i]] = [exp_acc[i]]
    else:
        studyexpMap[studies[i]].append(exp_acc[i])
        
# Generate data set with single cell type across different study, each study only take one experiment
# Containing cell type 'CL:2000001', which is 'group 1 innate lymphoid cell'
expPerStudy = []
keys = list(cellExpDict.keys())
studyList = []
for i in keys:
    if expstudyMap[i] not in studyList:
        studyList.append(expstudyMap[i])
        expPerStudy.append(i)
    else:
        continue

# Transform to the index
specific_cell_exp = expPerStudy
specific_cell_exp=set(specific_cell_exp)
# Build the Blood_Platelets exp expression matrix
specific_cell_exp_index = []
for i in range(len(exp_acc)):
    if exp_acc[i] in specific_cell_exp:
        specific_cell_exp_index.append(i)
    else:
        pass

specific_cell_exp_signature = get_signatures(specific_cell_exp_index)

# Unsure, according to the words, we should not include those gene with 0 m
temp_mean_x = np.mean(specific_cell_exp_signature,axis=1)
x = np.log(np.mean(specific_cell_exp_signature,axis=1)+1) # Confuse, ask later.

x = specific_cell_exp_signature[:,0]
y = np.std(specific_cell_exp_signature, axis=1)
for j in range(specific_cell_exp_signature.shape[1]):
    if j!=0:
        x = np.hstack((x,specific_cell_exp_signature[:,j]))
        y = np.hstack((y,np.std(specific_cell_exp_signature, axis=1)))

x = np.log(x+1)
y = np.log(y)

# Nan and Inf value needs to be dropped, otherwise we cannnot use Guassian KDE to estimate the density.
index = []
for i in range(len(y)):
    if np.isnan(y[i]) or np.isinf(y[i]):
        index.append(i)
    if x[i] == 0:
        index.append(i)
x = np.delete(x, index, axis= 0)
y = np.delete(y, index, axis= 0)

x_index = x.argsort()
estimatex, estimatey = x[x_index], y[x_index]

# Build a dictionary to record all x associated with y
xyDict = {}
for i in range(len(estimatex)):
    if estimatex[i] not in xyDict:
        xyDict[estimatex[i]] = [estimatey[i]]
    else:
        xyDict[estimatex[i]].append(estimatey[i])

# We randomly select a CV for corresponding expression level 
interpolatex = list(set(estimatex))
interpolatey = []
for i in interpolatex:
    temp = random.randint(0,len(xyDict[i]))-1
    interpolatey.append(xyDict[i][temp])
    
interpolatex = np.array(interpolatex)
interpolatey = np.array(interpolatey) 
idx = interpolatex.argsort()
x, y = interpolatex[idx], interpolatey[idx]
sp1 = UnivariateSpline(x, y,k=3)

print("Finished std-mean model training.")
print("Start training std-data model training.")

##############################
# Train the STD-Data Exp model
def get_signatures(index_list):
    """
    For a given list of indices in the file, returns a numpy array of
    gene signatures at those indices.
    """
    cpm = h5py.File(file_name, 'r')

    first = True
    for x in index_list:
        if first:
            signatures = cpm.get('cpm')[x]
            first = False
        else: 
            b = np.array(cpm.get('cpm')[x])
            signatures = np.column_stack([signatures, b])

    cpm.close()
    return signatures

# Provided h5 file
file_name = 'expression_CPM.h5'
decon_temp = './decon_temp/'
decon_temp_shell = "./decon_temp/"

# Load h5 data
cpm = h5py.File("expression_CPM.h5", 'r')
studies = np.array(cpm.get('study')).astype(str)
exp_acc = np.array(cpm.get('experiment_accession')).astype(str)
gene_ids = np.array(cpm.get('gene')).astype(str)
countspermillion = np.array(cpm.get('cpm'))
cpm.close()

with open('cell_types.json', 'r') as type_file:
    cell_type_file = json.load(type_file)
    
# Eliminate the redundant cell type in all exp
cell_type_specific_file = {}
for i in cell_type_file:
    cell_type_specific_file[i] = co.get_terms_without_children(cell_type_file[i])

cellExpDict = {}
for i in cell_type_specific_file:
    if cell_type_specific_file[i] == ['CL:2000001']:
        cellExpDict[i] = ['CL:2000001']

# Build the exp to study check dictionary
studyexpMap = {}
expstudyMap = {}
for i in range(len(exp_acc)):
    expstudyMap[exp_acc[i]] = studies[i]
    if studies[i] not in studyexpMap:
        studyexpMap[studies[i]] = [exp_acc[i]]
    else:
        studyexpMap[studies[i]].append(exp_acc[i])
        
# Generate data set with single cell type across different study, each study only take one experiment
# Containing cell type 'CL:2000001', which is 'group 1 innate lymphoid cell'
expPerStudy = []
keys = list(cellExpDict.keys())
studyList = []
for i in keys:
    if expstudyMap[i] not in studyList:
        studyList.append(expstudyMap[i])
        expPerStudy.append(i)
    else:
        continue

# Transform to the index
specific_cell_exp = expPerStudy
specific_cell_exp=set(specific_cell_exp)
# Build the Blood_Platelets exp expression matrix
specific_cell_exp_index = []
for i in range(len(exp_acc)):
    if exp_acc[i] in specific_cell_exp:
        specific_cell_exp_index.append(i)
    else:
        pass

specific_cell_exp_signature = get_signatures(specific_cell_exp_index)

# Unsure, according to the words, we should not include those gene with 0 m
temp_mean_x = np.mean(specific_cell_exp_signature,axis=1)
x = np.log(np.mean(specific_cell_exp_signature,axis=1)+1) # Confuse, ask later.

x = specific_cell_exp_signature[:,0]
y = np.std(specific_cell_exp_signature, axis=1)
for j in range(specific_cell_exp_signature.shape[1]):
    if j!=0:
        x = np.hstack((x,specific_cell_exp_signature[:,j]))
        y = np.hstack((y,np.std(specific_cell_exp_signature, axis=1)))

x = np.log(x+1)
y = np.log(y)

# Nan and Inf value needs to be dropped, otherwise we cannnot use Guassian KDE to estimate the density.
index = []
for i in range(len(y)):
    if np.isnan(y[i]) or np.isinf(y[i]):
        index.append(i)
    if x[i] == 0:
        index.append(i)
x = np.delete(x, index, axis= 0)
y = np.delete(y, index, axis= 0)

x_index = x.argsort()
estimatex, estimatey = x[x_index], y[x_index]

# Build a dictionary to record all x associated with y
xyDict = {}
for i in range(len(estimatex)):
    if estimatex[i] not in xyDict:
        xyDict[estimatex[i]] = [estimatey[i]]
    else:
        xyDict[estimatex[i]].append(estimatey[i])

# We randomly select a CV for corresponding expression level 
interpolatex = list(set(estimatex))
interpolatey = []
for i in interpolatex:
    temp = random.randint(0,len(xyDict[i]))-1
    interpolatey.append(xyDict[i][temp])
    
interpolatex = np.array(interpolatex)
interpolatey = np.array(interpolatey) 
idx = interpolatex.argsort()
x, y = interpolatex[idx], interpolatey[idx]
sp2 = UnivariateSpline(x, y,k=3)

print("Finished training std-data model.")
print("#################################")
print(" ")
print("Get Study List")
studieslist = []
for i in V6.get_studies():
    studieslist.append(i)

print("Start")

# Test for single data
# study_test = "ERP015294"
# testTwoDict = V6.decon_study(study_test, TM=True, TrainedNoiseModel_1 = sp1, TrainedNoiseModel_2 = sp2)
# print("Study: " + str(study_test))
# print("Normal: ")
# print(process_Normal.get_study_proportion(study_test))
# print("Weighted: ")
# print(process_Weight.get_study_proportion(study_test))
# print(" ")
# print(" ")

for i in studieslist:
    print(i)
    testTwoDict = V6.decon_study(i, TM=True, TrainedNoiseModel_1 = sp1, TrainedNoiseModel_2 = sp2)
    print("Study: " + str(i))
    print("Normal: ")
    print(process_Normal.get_study_proportion(i))
    print("Weighted: ")
    print(process_Weight.get_study_proportion(i))
    print(" ")
    print(" ")

print("Finished.")
