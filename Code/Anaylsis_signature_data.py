# import deconRNAseq
# import numpy as np
# import pandas as pd
# import h5py
# import json
# import subprocess
# import os.path
# import cell_ontology as co
# import time
# import datetime
# import os
# import re
# from scipy.optimize import curve_fit
# import random
# from scipy.interpolate import UnivariateSpline
# import matplotlib.pyplot as plt
# import pipeline2 as V5
# import process_tsvs_v2_Normal as process_Normal
# import process_tsvs_v2_Weight as process_Weight
# import math
# import matplotlib.pyplot as plt
# import statistics 
# import scipy
# import seaborn as sns
# from sklearn.metrics import mean_squared_error
# import importlib
# import matplotlib.patches as mpatches
# from itertools import repeat
# import scipy.cluster.hierarchy as sch
# import scipy.spatial.distance as csd
# import scipy.stats as ss
# from sklearn.metrics.pairwise import manhattan_distances
# import sklearn

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import os

def deconvolution_pipe_signature_build(cell_exp_count):
    # Read data set
    signature_pd = pd.read_csv('~/IndependentStudy/Data/SignatureSimulation/' + str(cell_exp_count) + '_signature.tsv', sep = '\t', index_col=0)
    signature_noise_np_pd = pd.read_csv('~/IndependentStudy/Data/SignatureSimulation/' + str(cell_exp_count) + '_signature_noise.tsv', sep = '\t', index_col=0)
    
    # Full gene set
    signature_noise_np_full = signature_noise_np_pd.to_numpy()
    signature_full = signature_pd.to_numpy()
    
    signature_noise_np_full = signature_noise_np_full.transpose()
    signature_full = signature_full.transpose()

    # Use touch to create new file
    os.system("touch " + "~/IndependentStudy/Data/Signature/signature_full_" + str(cell_exp_count) + ".tsv")
    os.system("touch " + "~/IndependentStudy/Data/NoisySignature/signature_full_"+ str(cell_exp_count) + ".tsv")

    np.savetxt("/ua/shi235/IndependentStudy/Data/Signature/signature_full_" + str(cell_exp_count) + ".tsv", signature_full, delimiter="\t")
    np.savetxt("/ua/shi235/IndependentStudy/Data/NoisySignature/signature_full_"+ str(cell_exp_count) + ".tsv", signature_noise_np_full, delimiter="\t")

    # Load ref_index (For refined gene)
    # path_name = ["~/IndependentStudy/Data/dtangle_500/", "~/IndependentStudy/Data/dtangle_001/" , "~/IndependentStudy/Data/dtangle_005/", "~/IndependentStudy/Data/dtangle_01/", "~/IndependentStudy/Data/dtangle_02/", "~/IndependentStudy/Data/dtangle_03/", "~/IndependentStudy/Data/dtangle_05/", "~/IndependentStudy/Data/dtangle_07/", "~/IndependentStudy/Data/dtangle_09/"]
    # ref_group = ['500', '001', '005', '01', '02', '03', '05', '07', '09']

    path_name = ["~/IndependentStudy/Data/dtangle_001/" , "~/IndependentStudy/Data/dtangle_005/", "~/IndependentStudy/Data/dtangle_01/", "~/IndependentStudy/Data/dtangle_02/", "~/IndependentStudy/Data/dtangle_03/", "~/IndependentStudy/Data/dtangle_05/", "~/IndependentStudy/Data/dtangle_07/", "~/IndependentStudy/Data/dtangle_09/"]
    ref_group = ['001', '005', '01', '02', '03', '05', '07', '09']

    # path_name = ["~/IndependentStudy/Data/dtangle_091/", "~/IndependentStudy/Data/dtangle_093/" , "~/IndependentStudy/Data/dtangle_095/", "~/IndependentStudy/Data/dtangle_097/", "~/IndependentStudy/Data/dtangle_099/"]
    # ref_group = ['091', '093', '095', '097', '099']
    # ref_group = ['0995', '0999', '1']

    ref_name = 'ref_dtangle_'
    ref_index_name = 'ref_index_'

    for i in range(len(ref_group)):
        print(i)
        # exec('ref_dtangle'  + " = pd.read_csv(\"~/IndependentStudy/Data/dtangle_" + ref_group[i] + "/" + str(cell_exp_count) + "_signature_filter.tsv\")")
        # exec('ref_index'  + " = list(ref_dtangle" + ".iloc[:,0])" )

        ref_dtangle = pd.read_csv("~/IndependentStudy/Data/dtangle_" + ref_group[i] + "/" + str(cell_exp_count) + "_signature_filter.tsv",)
        ref_index = list(ref_dtangle.iloc[:,0])
        print(ref_index)

        # Filter gene
        # Filter the gene using the marker gene
        signature_pd_filter = signature_pd.loc[:,ref_index]
        signature_noise_np_pd_filter = signature_noise_np_pd.loc[:,ref_index]

        # Back to np
        signature_noise_np_filter = signature_noise_np_pd_filter.to_numpy()
        signature_filter = signature_pd_filter.to_numpy()
        
        signature = signature_filter.transpose()
        signature_noise_np = signature_noise_np_filter.transpose()

        print(signature.shape)
        print(signature_noise_np.shape)
        
        # # Use touch to create new file
        os.system("touch " + "~/IndependentStudy/Data/Signature/signature_" + ref_group[i] + "_" + str(cell_exp_count) + ".tsv")
        os.system("touch " + "~/IndependentStudy/Data/NoisySignature/signature_" + ref_group[i] +  "_" + str(cell_exp_count) + ".tsv")

        np.savetxt("/ua/shi235/IndependentStudy/Data/Signature/signature_" + ref_group[i] + "_" + str(cell_exp_count) + ".tsv", signature, delimiter="\t")
        np.savetxt("/ua/shi235/IndependentStudy/Data/NoisySignature/signature_" + ref_group[i] +  "_" + str(cell_exp_count) + ".tsv", signature_noise_np, delimiter="\t")

        # Use touch to create new file
        # os.system("touch " + "/scratch/yujia/Signature/signature_" + ref_group[i] + "_" + str(cell_exp_count) + ".tsv")
        # os.system("touch " + "/scratch/yujia/NoisySignature/signature_" + ref_group[i] +  "_" + str(cell_exp_count) + ".tsv")

        # np.savetxt("/scratch/yujia/Signature/signature_" + ref_group[i] + "_" + str(cell_exp_count) + ".tsv", signature, delimiter="\t")
        # np.savetxt("/scratch/yujia/NoisySignature/signature_" + ref_group[i] +  "_" + str(cell_exp_count) + ".tsv", signature_noise_np, delimiter="\t")


def simulate_data_generation(cell_exp_count, dirichlet_para, ref_group, proportion):

    # noisy_signature = np.loadtxt("/scratch/yujia/NoisySignature/signature_" + ref_group + "_" + str(cell_exp_count) + ".tsv", delimiter="\t")
    noisy_signature = np.loadtxt("/ua/shi235/IndependentStudy/Data/NoisySignature/signature_" + ref_group + "_" + str(cell_exp_count) + ".tsv", delimiter="\t")
    
    print(noisy_signature.shape)
    data = np.matmul(noisy_signature, proportion)

    print(data.shape)
    print(" ")
    # save data
    # Use touch to create new file
    os.system("touch " + "~/IndependentStudy/Data/deconvoluteData/data_" + str(dirichlet_para) + "_" + ref_group + "_" + str(cell_exp_count) + ".tsv")

    np.savetxt("/ua/shi235/IndependentStudy/Data/deconvoluteData/data_" + str(dirichlet_para) + "_" + ref_group + "_" + str(cell_exp_count) + ".tsv" , data, delimiter="\t")

def pipeline(cell_exp_count):
    # ref_group = ['full','500', '001', '005', '01', '02', '03', '05', '07', '09']
    # ref_group = ['091', '093', '095', '097','099']
    # ref_group = ['0995', '0999', '1']
    ref_group = ['full', '001', '005', '01', '02', '03', '05', '07', '09']
    dirichlet_para = [0.001, 0.01, 0.1, 1]

    # propotion = np.random.dirichlet([j]*48)
    # # Use touch to create new file
    # os.system("touch " + "~/IndependentStudy/Data/trueProportion/proportion_" + str(dirichlet_para) + ".tsv")

    # np.savetxt("/ua/shi235/IndependentStudy/Data/trueProportion/proportion_" + str(dirichlet_para) + ".tsv", propotion, delimiter="\t")

    for j in dirichlet_para:
        # propotion = np.random.dirichlet([j]*48)
        propotion = np.loadtxt("/ua/shi235/IndependentStudy/Data/trueProportion/proportion_" + str(cell_exp_count) + "_" + str(j) + ".tsv", delimiter="\t")
        # Use touch to create new file
        # os.system("touch " + "~/IndependentStudy/Data/trueProportion/proportion_" + str(cell_exp_count) + "_" + str(j) + ".tsv")
        # np.savetxt("/ua/shi235/IndependentStudy/Data/trueProportion/proportion_" + str(cell_exp_count) + "_" + str(j) + ".tsv", propotion, delimiter="\t")
        for i in ref_group:
            simulate_data_generation(cell_exp_count, j, i, propotion)

if __name__ == "__main__":

    # Generate the signature
    Parallel(n_jobs=50)(
        delayed(deconvolution_pipe_signature_build)(exp) for exp in range(100)
    )

    print("Finished!")

    # Generate the data
    Parallel(n_jobs=50)(
        delayed(pipeline)(exp) for exp in range(100)
    )








