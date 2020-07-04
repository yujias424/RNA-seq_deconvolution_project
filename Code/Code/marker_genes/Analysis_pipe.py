# import deconRNAseq
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
from joblib import Parallel, delayed
import time

def multi_core(cell_exp_count, studies, exp_acc, gene_ids, countspermillion, qualified_cell_type_name, cell_type_file, qualified_cell_type):
    exp_acc_list = list(exp_acc)

    # Construct the noise added reference matrix
    reference_matrix = []
    select_study_list = {}
    for i in qualified_cell_type:
        tmp_exp = V5.celltype_to_exp(i)
        select_sample = random.choice(tmp_exp)
        select_study_list[i] = V5.exp_to_study(select_sample)
        exp_index = exp_acc_list.index(select_sample)
        reference_matrix.append(exp_index)

    # # Build the noise added reference matrix
    # for i in range(len(reference_matrix)):
    #     if i == 0:
    #         reference = countspermillion[reference_matrix[i]]
    #     else:
    #         tmp = countspermillion[reference_matrix[i]]
    #         reference = np.vstack((reference,tmp))

    # reference_noise = reference

    # # Build the reference matrix
    # reference_noise_free = []
    # for i in range(len(qualified_cell_type)):
    #     tmp_exp = V5.celltype_to_exp(qualified_cell_type[i])
    #     tmp_ref = []
    #     # Since all cell type will be included, therefore we can simply using the previous one`
    #     for j in tmp_exp:
    #         # Only one study wll be chosen to construct the noisy reference, therefore using !=
    #         if V5.exp_to_study(j) != select_study_list[qualified_cell_type[i]]: 
    #             tmp_ref.append(exp_acc_list.index(j))

    #     for j in range(len(tmp_ref)):
    #         if j == 0:
    #             reference = countspermillion[tmp_ref[j]]
    #         else:
    #             tmp = countspermillion[tmp_ref[j]]
    #             reference = np.vstack((reference,tmp))

    #     if len(tmp_ref) > 1:
    #         ref_mean = np.mean(reference, axis=0)
    #     else:
    #         ref_mean = reference
        
    #     reference_noise_free.append(ref_mean)

    # reference_noise_free_np = np.array(reference_noise_free)
    # signature_np = np.transpose(reference_noise_free_np)
    # reference_noise_np = reference_noise.copy()
    # signature_noise_np = np.transpose(reference_noise_np)
    # # signature_temp = signature_np.copy()
    
    # # Transform to pandas
    # signature_np = signature_np.transpose()
    # signature_noise_np = signature_noise_np.transpose()
    
    # signature_pd = pd.DataFrame(data=signature_np, columns=gene_ids, index=qualified_cell_type_name)
    # signature_noise_np_pd = pd.DataFrame(data = signature_noise_np
    #                                      , columns=gene_ids, index=[co.get_term_name(i) for i in qualified_cell_type])
    
    # Save the signature and noisy signature for future analysis
    # signature_pd.to_csv('~/IndependentStudy/Data/SignatureSimulation/' + str(cell_exp_count) + '_signature.tsv', sep = '\t')
    # signature_noise_np_pd.to_csv('~/IndependentStudy/Data/SignatureSimulation/' + str(cell_exp_count) + '_signature_noise.tsv', sep = '\t')

    # Build the variance data set
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

    # Build the variance matrix
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
        print(keys)
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



if __name__ == '__main__':
    # Load CPM file
    cpm = h5py.File("expression_CPM.h5", 'r')
    studies = np.array(cpm.get('study')).astype(str)
    exp_acc = np.array(cpm.get('experiment_accession')).astype(str)
    gene_ids = np.array(cpm.get('gene')).astype(str)
    countspermillion = np.array(cpm.get('cpm'))
    cpm.close()

    qualified_cell_type = ['CL:1000274', 'CL:0002618', 'CL:0000501', 'CL:0000765', 'CL:2000001', 'CL:0002341', 'CL:0000583', 'CL:0000127', 'CL:0002631', 'CL:0000936', 'CL:0002327', 'CL:0000023', 'CL:0000216', 'CL:0000557', 'CL:0000018', 'CL:0000905', 'CL:0000182', 'CL:0000895', 'CL:0000096', 'CL:0002340', 'CL:0011001', 'CL:0000050', 'CL:0002633', 'CL:0000232', 'CL:0000019', 'CL:0000792', 'CL:0002063', 'CL:0000836', 'CL:0000904', 'CL:0002399', 'CL:0000233', 'CL:0002038', 'CL:0000788', 'CL:0000900', 'CL:0000670', 'CL:0002057', 'CL:0000351', 'CL:0001069', 'CL:0000091', 'CL:0000359', 'CL:0010004', 'CL:0000171', 'CL:0000169', 'CL:0000017', 'CL:0000623', 'CL:0001057', 'CL:0002394', 'CL:0000129']
    qualified_cell_type_name = [co.get_term_name(i) for i in qualified_cell_type]

    with open('cell_types.json', 'r') as type_file:
        cell_type_file = json.load(type_file)

    print("Start!")
    now = time.time()
    Parallel(n_jobs=1)(
        delayed(multi_core)(i, studies, exp_acc, gene_ids, countspermillion, qualified_cell_type_name, cell_type_file, qualified_cell_type) for i in range(1)
    )
    # for i in range(5):
    #     print(i)
    #     multi_core(i, studies, exp_acc, gene_ids, countspermillion, qualified_cell_type_name)
    print("Finished in", time.time()-now , "sec")
# Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))

# 1264.1261851787567