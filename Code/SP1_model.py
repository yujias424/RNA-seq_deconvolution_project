import deconRNAseq
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
import pipeline2 as V5
import process_tsvs_v2_Normal as process_Normal
import process_tsvs_v2_Weight as process_Weight
import math
import matplotlib.pyplot as plt
import statistics 
import scipy
import seaborn as sns
from sklearn.metrics import mean_squared_error
import importlib
import matplotlib.patches as mpatches
from itertools import repeat
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as csd
import scipy.stats as ss
from sklearn.metrics.pairwise import manhattan_distances
import sklearn

def get_signatures(index_list):
    """
    For a given list of indices in the file, returns a numpy array of
    gene signatures at those indices.
    """
    file_name = 'expression_CPM.h5'
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

def build_model():
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

    # print(cellExpDict)
    # Build the exp to study check dictionary
    studyexpMap = {}
    expstudyMap = {}
    for i in range(len(exp_acc)):
        expstudyMap[exp_acc[i]] = studies[i]
        if studies[i] not in studyexpMap:
            studyexpMap[studies[i]] = [exp_acc[i]]
        else:
            studyexpMap[studies[i]].append(exp_acc[i])

    # print(studyexpMap)
    # Generate data set with single cell type across different study, each study only take one experiment
    # Containing cell type 'CL:0001067', which is 'group 1 innate lymphoid cell'
    expPerStudy = []
    keys = list(cellExpDict.keys())
    studyList = []
    for i in keys:
        if expstudyMap[i] not in studyList:
            studyList.append(expstudyMap[i])
            expPerStudy.append(i)
        else:
            continue

    # print(len(keys))
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

    # print(len(specific_cell_exp))
    specific_cell_exp_signature = get_signatures(specific_cell_exp_index)

    # Unsure, according to the words, we should not include those gene with 0 m
    temp_mean_x = np.mean(specific_cell_exp_signature,axis=1)

    x = np.log(np.mean(specific_cell_exp_signature,axis=1)+1) # Confuse, ask later.
    y = np.log(np.std(specific_cell_exp_signature, axis=1))
    # print(len(x))
    # print(x)
    # print(y)

    # Nan and Inf value needs to be dropped, otherwise we cannnot use Guassian KDE to estimate the density.
    index = []
    for i in range(len(y)):
    #     if np.isnan(y[i]) or np.isinf(y[i]):
    #         index.append(i)
        if x[i] == 0:
            index.append(i)
    x = np.delete(x, index, axis= 0)
    y = np.delete(y, index, axis= 0)
    # print(len(x))
    # print(x)
    # print(y)

    fig, ax1= plt.subplots()
    ax1.scatter(x, y, alpha=0.6)
    ax1.set_title("Std-Mean Plot in Log Space")
    ax1.set_ylabel("log(STD)")
    ax1.set_xlabel("log(MeanCPM+1)")
    # plt.savefig("CV_mean_sp1.png")

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
    sp1 = UnivariateSpline(x, y,k=1)

    return sp1

def build_model_cv_mean():
    
    # Gene Specific model
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
        
    # Build the exp to study check dictionary
    studyexpMap = {}
    expstudyMap = {}
    for i in range(len(exp_acc)):
        expstudyMap[exp_acc[i]] = studies[i]
        if studies[i] not in studyexpMap:
            studyexpMap[studies[i]] = [exp_acc[i]]
        else:
            studyexpMap[studies[i]].append(exp_acc[i])

    # Only focus on the selected 48 cell types
    cell_types_selected = ['CL:1000274', 'CL:0002618', 'CL:0000501', 'CL:0000765', 'CL:2000001', 'CL:0002341', 'CL:0000583', 'CL:0000127', 'CL:0002631', 'CL:0000936', 'CL:0002327', 'CL:0000023', 'CL:0000216', 'CL:0000557', 'CL:0000018', 'CL:0000905', 'CL:0000182', 'CL:0000895', 'CL:0000096', 'CL:0002340', 'CL:0011001', 'CL:0000050', 'CL:0002633', 'CL:0000232', 'CL:0000019', 'CL:0000792', 'CL:0002063', 'CL:0000836', 'CL:0000904', 'CL:0002399', 'CL:0000233', 'CL:0002038', 'CL:0000788', 'CL:0000900', 'CL:0000670', 'CL:0002057', 'CL:0000351', 'CL:0001069', 'CL:0000091', 'CL:0000359', 'CL:0010004', 'CL:0000171', 'CL:0000169', 'CL:0000017', 'CL:0000623', 'CL:0001057', 'CL:0002394', 'CL:0000129']

    cell_types_48 = []

    # Learn CV from specific cell types
    for cell_co in cell_types_selected:

        cellExpDict = {}
        for i in cell_type_specific_file:
            if cell_co in cell_type_specific_file[i]:
                cellExpDict[i] = [cell_co]
                
        # cell type specific Exp to Study dictionary
        expPerStudy = []
        keys = list(cellExpDict.keys())
        studyList = []
        for i in keys:
            if expstudyMap[i] not in studyList:
                studyList.append(expstudyMap[i])
                expPerStudy.append(i)
            else:
                continue
                
        tmp_exp_study = {}
        for i in cellExpDict.keys():
            if expstudyMap[i] not in tmp_exp_study.keys():
                tmp_exp_study[expstudyMap[i]] = [i]
            else:
                tmp_exp_study[expstudyMap[i]].append(i)

        # Generate the mean profile
        tmp_mean = []
        # Build the Blood_Platelets exp expression matrix
        for j in tmp_exp_study.items():
            
            # Garb the cell index
            specific_cell_exp_index = []
            for i in range(len(exp_acc)):
                if exp_acc[i] in j[1]:
                    specific_cell_exp_index.append(i)
                else:
                    continue

            specific_cell_exp_signature = get_signatures(specific_cell_exp_index)
            
            # Generate the cell_type specific mean
            if len(j[1]) == 1:
                tmp_mean.append(specific_cell_exp_signature)
            else:
                tmp_mean.append(np.mean(specific_cell_exp_signature, axis=1))
        
        cell_types_48 += tmp_mean

    cell_types_48 = np.array(cell_types_48)

    x = np.log(np.mean(cell_types_48,axis=0)+1) # Confuse, ask later.
    y = np.log(ss.variation(cell_types_48, axis=0))

    #   Nan and Inf value needs to be dropped, otherwise we cannnot use Guassian KDE to estimate the density.
    index = []
    for i in range(len(y)):
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
    sp1 = UnivariateSpline(x, y, k=1)

    return sp1

def cell_type_variance():

    # Gene Specific model
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
        
    # Build the exp to study check dictionary
    studyexpMap = {}
    expstudyMap = {}
    for i in range(len(exp_acc)):
        expstudyMap[exp_acc[i]] = studies[i]
        if studies[i] not in studyexpMap:
            studyexpMap[studies[i]] = [exp_acc[i]]
        else:
            studyexpMap[studies[i]].append(exp_acc[i])

    cell_types_selected = ['CL:1000274', 'CL:0002618', 'CL:0000501', 'CL:0000765', 'CL:2000001', 'CL:0002341', 'CL:0000583', 'CL:0000127', 'CL:0002631', 'CL:0000936', 'CL:0002327', 'CL:0000023', 'CL:0000216', 'CL:0000557', 'CL:0000018', 'CL:0000905', 'CL:0000182', 'CL:0000895', 'CL:0000096', 'CL:0002340', 'CL:0011001', 'CL:0000050', 'CL:0002633', 'CL:0000232', 'CL:0000019', 'CL:0000792', 'CL:0002063', 'CL:0000836', 'CL:0000904', 'CL:0002399', 'CL:0000233', 'CL:0002038', 'CL:0000788', 'CL:0000900', 'CL:0000670', 'CL:0002057', 'CL:0000351', 'CL:0001069', 'CL:0000091', 'CL:0000359', 'CL:0010004', 'CL:0000171', 'CL:0000169', 'CL:0000017', 'CL:0000623', 'CL:0001057', 'CL:0002394', 'CL:0000129']

    variance_matrix = []
    cell_types_48 = []
    for cell_co in cell_types_selected:
        # Get the cell type
        cellExpDict = {}
        for i in cell_type_specific_file:
            if cell_co in cell_type_specific_file[i]:
                cellExpDict[i] = [cell_co]
                
        # cell type specific Exp to Study dictionary
        expPerStudy = []
        keys = list(cellExpDict.keys())
        studyList = []
        for i in keys:
            if expstudyMap[i] not in studyList:
                studyList.append(expstudyMap[i])
                expPerStudy.append(i)
            else:
                continue
                
        tmp_exp_study = {}
        for i in cellExpDict.keys():
            if expstudyMap[i] not in tmp_exp_study.keys():
                tmp_exp_study[expstudyMap[i]] = [i]
            else:
                tmp_exp_study[expstudyMap[i]].append(i)
                
        
        # Get the within study variance
        # Generate the mean profile
        tmp_mean = []
        within_study_var = []
        # Build the exp expression matrix
        for j in tmp_exp_study.items():
            
            # Garb the cell index
            specific_cell_exp_index = []
            for i in range(len(exp_acc)):
                if exp_acc[i] in j[1]:
                    specific_cell_exp_index.append(i)
                else:
                    continue

            specific_cell_exp_signature = get_signatures(specific_cell_exp_index)
            
            # Generate the cell_type specific mean (j[1] is a tuple), tmp_mean consist study mean 
            if len(j[1]) == 1:
                tmp_mean.append(specific_cell_exp_signature)
            else:
                tmp_mean.append(np.mean(specific_cell_exp_signature, axis=1))
                
            # Calculate the residue (if j[1] > 1)
            if len(j[1]) > 1:
                tmp_residue_list = []
                for index in specific_cell_exp_index:
                    tmp_exp = get_signatures([index])
                    tmp_residue = np.abs(tmp_exp - np.mean(specific_cell_exp_signature, axis=1))
                    tmp_residue_list.append(tmp_residue)

                # Construct the within study variance
                tmp_residue_list = np.array(tmp_residue_list)
                within_study_var.append(np.var(tmp_residue_list, axis=0))
            else:
                within_study_var.append(np.zeros(specific_cell_exp_signature.shape[0]))
        
        cell_types_48 += tmp_mean
        within_study_var = np.array(within_study_var)
        
        # Construct the study variance
        tmp_mean = np.array(tmp_mean)
        study_variance = np.var(tmp_mean, axis=0)
        
        # We assume variance sum law here
        total_variance = np.zeros(study_variance.shape[0])
        total_variance = total_variance + study_variance
        
        for i in within_study_var:
            total_variance = total_variance + i

        variance_matrix.append(total_variance)

    variance_matrix = np.array(variance_matrix)
    # print(variance_matrix.shape)
    # return np.sum(variance_matrix, axis=0)
    return variance_matrix

def cell_type_variance_model():

    # Gene Specific model
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
        
    # Build the exp to study check dictionary
    studyexpMap = {}
    expstudyMap = {}
    for i in range(len(exp_acc)):
        expstudyMap[exp_acc[i]] = studies[i]
        if studies[i] not in studyexpMap:
            studyexpMap[studies[i]] = [exp_acc[i]]
        else:
            studyexpMap[studies[i]].append(exp_acc[i])

    cell_types_selected = ['CL:1000274', 'CL:0002618', 'CL:0000501', 'CL:0000765', 'CL:2000001', 'CL:0002341', 'CL:0000583', 'CL:0000127', 'CL:0002631', 'CL:0000936', 'CL:0002327', 'CL:0000023', 'CL:0000216', 'CL:0000557', 'CL:0000018', 'CL:0000905', 'CL:0000182', 'CL:0000895', 'CL:0000096', 'CL:0002340', 'CL:0011001', 'CL:0000050', 'CL:0002633', 'CL:0000232', 'CL:0000019', 'CL:0000792', 'CL:0002063', 'CL:0000836', 'CL:0000904', 'CL:0002399', 'CL:0000233', 'CL:0002038', 'CL:0000788', 'CL:0000900', 'CL:0000670', 'CL:0002057', 'CL:0000351', 'CL:0001069', 'CL:0000091', 'CL:0000359', 'CL:0010004', 'CL:0000171', 'CL:0000169', 'CL:0000017', 'CL:0000623', 'CL:0001057', 'CL:0002394', 'CL:0000129']

    variance_matrix = []
    cell_types_48 = []
    for cell_co in cell_types_selected:
        # Get the cell type
        cellExpDict = {}
        for i in cell_type_specific_file:
            if cell_co in cell_type_specific_file[i]:
                cellExpDict[i] = [cell_co]
                
        # cell type specific Exp to Study dictionary
        expPerStudy = []
        keys = list(cellExpDict.keys())
        studyList = []
        for i in keys:
            if expstudyMap[i] not in studyList:
                studyList.append(expstudyMap[i])
                expPerStudy.append(i)
            else:
                continue
                
        tmp_exp_study = {}
        for i in cellExpDict.keys():
            if expstudyMap[i] not in tmp_exp_study.keys():
                tmp_exp_study[expstudyMap[i]] = [i]
            else:
                tmp_exp_study[expstudyMap[i]].append(i)
                
        
        # Get the within study variance
        # Generate the mean profile
        tmp_mean = []
        within_study_var = []
        # Build the exp expression matrix
        for j in tmp_exp_study.items():
            
            # Garb the cell index
            specific_cell_exp_index = []
            for i in range(len(exp_acc)):
                if exp_acc[i] in j[1]:
                    specific_cell_exp_index.append(i)
                else:
                    continue

            specific_cell_exp_signature = get_signatures(specific_cell_exp_index)
            
            # Generate the cell_type specific mean (j[1] is a tuple), tmp_mean consist study mean 
            if len(j[1]) == 1:
                tmp_mean.append(specific_cell_exp_signature)
            else:
                tmp_mean.append(np.mean(specific_cell_exp_signature, axis=1))
                
            # Calculate the residue (if j[1] > 1)
            if len(j[1]) > 1:
                tmp_residue_list = []
                for index in specific_cell_exp_index:
                    tmp_exp = get_signatures([index])
                    tmp_residue = np.abs(tmp_exp - np.mean(specific_cell_exp_signature, axis=1))
                    tmp_residue_list.append(tmp_residue)

                # Construct the within study variance
                tmp_residue_list = np.array(tmp_residue_list)
                within_study_var.append(np.var(tmp_residue_list, axis=0))
            else:
                within_study_var.append(np.zeros(specific_cell_exp_signature.shape[0]))
        
        cell_types_48 += tmp_mean
        within_study_var = np.array(within_study_var)
        
        # Construct the study variance
        tmp_mean = np.array(tmp_mean)
        study_variance = np.var(tmp_mean, axis=0)
        
        # We assume variance sum law here
        total_variance = np.zeros(study_variance.shape[0])
        total_variance = total_variance + study_variance
        
        for i in within_study_var:
            total_variance = total_variance + i

        variance_matrix.append(total_variance)

    variance_matrix = np.array(variance_matrix)

    variance_matrix_sum = np.sum(variance_matrix, axis=0)

    x = np.log(np.mean(cell_types_48,axis=0)+1) # Confuse, ask later.
    y = np.log(np.sqrt(variance_matrix_sum)+1)

    #   Nan and Inf value needs to be dropped, otherwise we cannnot use Guassian KDE to estimate the density.
    index = []
    for i in range(len(y)):
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
    sp1 = UnivariateSpline(x, y, k=5)

    return sp1


if __name__ == "__main__":
    pass