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
# from joblib import Parallel, delayed

# import deconRNAseq
import deconRNAseq_fardeep as deconRNAseq
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import variation
import SP1_model
# import cell_ontology as co
import os

def nnls(exp_index):
    # ref_group = ['full','500', '001', '005', '01', '02', '03', '05', '07', '09']
    # dirichlet_para = [0.001, 0.01, 0.1, 1]

    # ref_group = ['091', '093', '095','097', '099']
    # ref_group = ['0995', '0999', '1']
    ref_group = ['full', '001', '005', '01', '02', '03', '05', '07', '09']
    dirichlet_para = [0.001, 0.01, 0.1, 1]

    for i in ref_group:
        signature = np.loadtxt('/ua/shi235/IndependentStudy/Data/Signature/signature_' + i + "_" + str(exp_index) + ".tsv", delimiter="\t")
        for j in dirichlet_para:
            data = np.loadtxt('/ua/shi235/IndependentStudy/Data/deconvoluteData/data_' + str(j) +  "_" +i + "_" + str(exp_index) + ".tsv", delimiter="\t")
            result = deconRNAseq.deconrnaseqweight(data, signature, deconMethod = 'normal', addone = True)[0][0]
            # print(result)
            np.savetxt("/ua/shi235/IndependentStudy/Data/EstimatedResult/nnls_2/proportion_" + str(exp_index) + "_" + str(j) + "_" + i + ".tsv", result)
        #     break
        # break

def weightedIRLS(exp_index, model):
    # ref_group = ['full','500', '001', '005', '01', '02', '03', '05', '07', '09']
    ref_group = ['full', '001', '005', '01', '02', '03', '05', '07', '09']
    dirichlet_para = [0.001, 0.01, 0.1, 1]
    model = np.loadtxt('/ua/shi235/IndependentStudy/Data/Variance/' + str(exp_index) + '_variance.txt', delimiter="\t")
    
    # ref_group = ['091', '093', '095','097', '099']
    # ref_group = ['0995', '0999', '1']
    # dirichlet_para = [0.001, 0.01, 0.1, 1]

    # Load ref_index (For refined gene)
    path_name = ["~/IndependentStudy/Data/dtangle_500/", "~/IndependentStudy/Data/dtangle_001/" , "~/IndependentStudy/Data/dtangle_005/", "~/IndependentStudy/Data/dtangle_01/", "~/IndependentStudy/Data/dtangle_02/", "~/IndependentStudy/Data/dtangle_03/", "~/IndependentStudy/Data/dtangle_05/", "~/IndependentStudy/Data/dtangle_07/", "~/IndependentStudy/Data/dtangle_09/"]
    ref_group_1 = ['500', '001', '005', '01', '02', '03', '05', '07', '09']

    for i in range(len(ref_group)):
        signature = np.loadtxt('/ua/shi235/IndependentStudy/Data/Signature/signature_' + ref_group[i] + "_" + str(exp_index) + ".tsv", delimiter="\t")
        
        # Learn the coefficient of variation of from the signature matrix.
        # print(signature.shape)
        # CV_sig = variation(signature+1, axis = 1)
        # print(CV_sig.shape)

        if ref_group[i] in ref_group_1:
            ref_dtangle = pd.read_csv("~/IndependentStudy/Data/dtangle_" + ref_group[i] + "/" + str(exp_index) + "_signature_filter.tsv",)
            ref_index = list(ref_dtangle.iloc[:,1])
            model_select = np.take(model.copy(), ref_index, axis=1)
        else:
            model_select = model.copy()

        # print(model_select.shape)
        # print(signature.shape)
            # print(ref_index)
        # print(len(model_select))
        for j in dirichlet_para:
            data = np.loadtxt('/ua/shi235/IndependentStudy/Data/deconvoluteData/data_' + str(j) +  "_" +ref_group[i] + "_" + str(exp_index) + ".tsv", delimiter="\t")
            # CV_sig = np.exp(model(np.log(data+1)))
            # print(CV_sig.shape)
            # IRLS
            proportion_start = deconRNAseq.deconrnaseqweight(data, signature, true_std = None
                                        , deconMethod = 'normal', ProvideTM = None
                                        , addone = True, drop_low = True, olrZero=True)[0][0]
            
            for q in range(10):
                proportion_start1 = proportion_start
                proportion_start = deconRNAseq.deconrnaseqweightIRLS(data, signature, true_std = np.transpose(signature)
                                            , deconMethod = 'std_prop', ProvideTM = None
                                            , addone = True, drop_low = True, olrZero = False
                                            , drop_gene=False, threshold=0, Proportion=np.array(proportion_start1))[0][0]

            result = deconRNAseq.deconrnaseqweightIRLS(data, signature, true_std = np.transpose(signature)
                                        , deconMethod = 'std_prop', ProvideTM = None
                                        , addone = True, drop_low = True, olrZero = False
                                        , drop_gene=False, threshold=0, Proportion=proportion_start)[0][0]

            # print(result)
            # np.savetxt("/ua/shi235/IndependentStudy/Data/EstimatedResult/irls/proportion_" + str(exp_index) + "_" + str(j) + "_" + i + ".tsv", result)
            # os.system("touch " + "~/IndependentStudy/Data/nnls_irls_var_allexp/proportion_" + str(exp_index) + "_" + str(j) + "_" + ref_group[i] + ".tsv")

            np.savetxt("/ua/shi235/IndependentStudy/Data/EstimatedResult/nnls_irls_var_sig/proportion_" + str(exp_index) + "_" + str(j) + "_" + ref_group[i] + ".tsv", result)
        
        #     break
        # break

def dwls(exp_index):
    qualified_cell_type = ['CL:1000274', 'CL:0002618', 'CL:0000501', 'CL:0000765', 'CL:2000001', 'CL:0002341', 'CL:0000583', 'CL:0000127', 'CL:0002631', 'CL:0000936', 'CL:0002327', 'CL:0000023', 'CL:0000216', 'CL:0000557', 'CL:0000018', 'CL:0000905', 'CL:0000182', 'CL:0000895', 'CL:0000096', 'CL:0002340', 'CL:0011001', 'CL:0000050', 'CL:0002633', 'CL:0000232', 'CL:0000019', 'CL:0000792', 'CL:0002063', 'CL:0000836', 'CL:0000904', 'CL:0002399', 'CL:0000233', 'CL:0002038', 'CL:0000788', 'CL:0000900', 'CL:0000670', 'CL:0002057', 'CL:0000351', 'CL:0001069', 'CL:0000091', 'CL:0000359', 'CL:0010004', 'CL:0000171', 'CL:0000169', 'CL:0000017', 'CL:0000623', 'CL:0001057', 'CL:0002394', 'CL:0000129']
    # ref_group = ['full','500', '001', '005', '01', '02', '03', '05', '07', '09']
    # ref_group = ['01', '02', '03', '05', '07', '09',"full"]
    # ref_group = ['500', '001', '005']
    ref_group = ['01', '02', '03', '05']
    dirichlet_para = [0.001, 0.01, 0.1, 1]

    for i in ref_group:
        signature = np.loadtxt('/ua/shi235/IndependentStudy/Data/Signature/signature_' + i + "_" + str(exp_index) + ".tsv", delimiter="\t")
        for j in dirichlet_para:
            # print(j)
            data = np.loadtxt('/ua/shi235/IndependentStudy/Data/deconvoluteData/data_' + str(j) +  "_" +i + "_" + str(exp_index) + ".tsv", delimiter="\t")
            
            Gene_symbol = np.arange(0,data.reshape((1,len(data))).shape[1])
            a = pd.DataFrame(data, index=Gene_symbol)
            a.to_csv('/ua/shi235/IndependentStudy/Data/tempdata/dwls/' + str(exp_index) + "/mixture_dwls.txt", sep='\t')

            # Signature
            signature = pd.DataFrame(signature, index=Gene_symbol, columns= qualified_cell_type)
            signature.to_csv('/ua/shi235/IndependentStudy/Data/tempdata/dwls/' + str(exp_index) + "/dwls_signature.txt", sep='\t')
        
            # Run
            # os.system('~/conda_env/py36/bin/Rscript fardeep_script.r')
            os.system('Rscript ' + '/ua/shi235/IndependentStudy/Data/tempdata/dwls/' + str(exp_index) + "/dwls_script.r")

            # Result
            # result = pd.read_csv('/ua/shi235/IndependentStudy/Data/tempdata/dwls/' + str(exp_index) + "/dwls_result.txt", index_col=0).iloc[1,0:48].to_numpy()

            result = pd.read_csv('/ua/shi235/IndependentStudy/Data/tempdata/dwls/' + str(exp_index) + "/dwls_result.txt", index_col=0).to_numpy().transpose()[0]

            # print(result)

            np.savetxt("/ua/shi235/IndependentStudy/Data/EstimatedResult/dwls/proportion_" + str(exp_index) + "_" + str(j) + "_" + i + ".tsv", result)
        #     break
        # break


def dtangle(exp_index):
    ref_group = ['full']
    dirichlet_para = [0.001, 0.01, 0.1, 1]

    for i in ref_group:
        signature = np.loadtxt('/ua/shi235/IndependentStudy/Data/Signature/signature_' + i + "_" + str(exp_index) + ".tsv", delimiter="\t")
        # print(signature.shape)
        # Signature
        Gene_symbol = np.arange(0,signature.shape[0])
        signature = pd.DataFrame(signature.transpose(), columns=Gene_symbol)
        signature_dtangle = np.log2(signature+1)
        signature_dtangle.to_csv('/ua/shi235/IndependentStudy/Data/tempdata/dtangle/' + str(exp_index) + '/signature_dtangle.txt', sep='\t')
        for j in dirichlet_para:
            print(j)
            data = np.loadtxt('/ua/shi235/IndependentStudy/Data/deconvoluteData/data_' + str(j) +  "_" +i + "_" + str(exp_index) + ".tsv", delimiter="\t")

            # Mixture
            Gene_symbol = np.arange(0,data.reshape((1,len(data))).shape[1])
            a = pd.DataFrame(data.reshape((1,len(data))), columns=Gene_symbol)
            df = np.log2(a+0.1)
            df.to_csv('/ua/shi235/IndependentStudy/Data/tempdata/dtangle/' + str(exp_index) + '/mixture_dtangle.txt', sep='\t')

            prop_taken = ['02','03','05','07','09']

            for pk in prop_taken:
                os.system('Rscript ' + '/ua/shi235/IndependentStudy/Data/tempdata/dtangle/' + str(exp_index) + "/rscript_"  + pk + ".r")
                dtangle_result = pd.read_csv('/ua/shi235/IndependentStudy/Data/tempdata/dtangle/' + str(exp_index) + '/dtangle_result.txt', sep=',', index_col=0)
                result = np.array(list(dtangle_result.iloc[:,0])) # Only one mixture process each iteration
                np.savetxt("/ua/shi235/IndependentStudy/Data/EstimatedResult/dtangle/" + pk + "_result/proportion_" + str(exp_index) + "_" + str(j) + "_" + i + ".tsv", result)
                # break
            # 10% top genes (Default)
            # os.system('Rscript ' + '/ua/shi235/IndependentStudy/Data/tempdata/dtangle/' + str(exp_index) + '/rscript.r')

            # 1% top genes
            # os.system('Rscript ' + '/ua/shi235/IndependentStudy/Data/tempdata/dtangle/' + str(exp_index) + '/rscript_1.r')

            # dtangle_result = pd.read_csv('/ua/shi235/IndependentStudy/Data/tempdata/dtangle/' + str(exp_index) + '/dtangle_result.txt', sep=',', index_col=0)
            # result = np.array(list(dtangle_result.iloc[:,0])) # Only one mixture process each iteration
            # print(result)

            # 10% top genes (Default)
            # np.savetxt("/ua/shi235/IndependentStudy/Data/EstimatedResult/dtangle/proportion_" + str(exp_index) + "_" + str(j) + "_" + i + ".tsv", result)

            # 1% top genes
            # np.savetxt("/ua/shi235/IndependentStudy/Data/EstimatedResult/dtangle/001_result/proportion_" + str(exp_index) + "_" + str(j) + "_" + i + ".tsv", result)
            # break
        # break

def fardeep(exp_index):
    qualified_cell_type = ['CL:1000274', 'CL:0002618', 'CL:0000501', 'CL:0000765', 'CL:2000001', 'CL:0002341', 'CL:0000583', 'CL:0000127', 'CL:0002631', 'CL:0000936', 'CL:0002327', 'CL:0000023', 'CL:0000216', 'CL:0000557', 'CL:0000018', 'CL:0000905', 'CL:0000182', 'CL:0000895', 'CL:0000096', 'CL:0002340', 'CL:0011001', 'CL:0000050', 'CL:0002633', 'CL:0000232', 'CL:0000019', 'CL:0000792', 'CL:0002063', 'CL:0000836', 'CL:0000904', 'CL:0002399', 'CL:0000233', 'CL:0002038', 'CL:0000788', 'CL:0000900', 'CL:0000670', 'CL:0002057', 'CL:0000351', 'CL:0001069', 'CL:0000091', 'CL:0000359', 'CL:0010004', 'CL:0000171', 'CL:0000169', 'CL:0000017', 'CL:0000623', 'CL:0001057', 'CL:0002394', 'CL:0000129']
    # ref_group = ['full','500', '001', '005', '01', '02', '03', '05', '07', '09']
    ref_group = ['500', '001', '005', '01', '02', '03', '05', '07', '09',"full"]
    dirichlet_para = [0.001, 0.01, 0.1, 1]

    for i in ref_group:
        signature = np.loadtxt('/ua/shi235/IndependentStudy/Data/Signature/signature_' + i + "_" + str(exp_index) + ".tsv", delimiter="\t")
        for j in dirichlet_para:
            # print(j)
            data = np.loadtxt('/ua/shi235/IndependentStudy/Data/deconvoluteData/data_' + str(j) +  "_" +i + "_" + str(exp_index) + ".tsv", delimiter="\t")
            
            Gene_symbol = np.arange(0,data.reshape((1,len(data))).shape[1])
            a = pd.DataFrame(data, index=Gene_symbol)
            a.to_csv('/ua/shi235/IndependentStudy/Data/tempdata/mixture_fardeep.txt', sep='\t')

            # Signature
            signature = pd.DataFrame(signature, index=Gene_symbol, columns= qualified_cell_type)
            signature.to_csv('/ua/shi235/IndependentStudy/Data/tempdata/fardeep_signature.tsv', sep='\t')
        
            # Run
            os.system('~/conda_env/py36/bin/Rscript fardeep_script.r')

            # Result
            result = pd.read_csv('/ua/shi235/IndependentStudy/Data/tempdata/fardeep_result.csv', index_col=0).iloc[1,0:48].to_numpy()
            # print(result)
            np.savetxt("/ua/shi235/IndependentStudy/Data/EstimatedResult/fardeep/proportion_" + str(exp_index) + "_" + str(j) + "_" + i + ".tsv", result)
        #     break
        # break
    # print("Finished.")

def cibersort(exp_index):
    qualified_cell_type = ['CL:1000274', 'CL:0002618', 'CL:0000501', 'CL:0000765', 'CL:2000001', 'CL:0002341', 'CL:0000583', 'CL:0000127', 'CL:0002631', 'CL:0000936', 'CL:0002327', 'CL:0000023', 'CL:0000216', 'CL:0000557', 'CL:0000018', 'CL:0000905', 'CL:0000182', 'CL:0000895', 'CL:0000096', 'CL:0002340', 'CL:0011001', 'CL:0000050', 'CL:0002633', 'CL:0000232', 'CL:0000019', 'CL:0000792', 'CL:0002063', 'CL:0000836', 'CL:0000904', 'CL:0002399', 'CL:0000233', 'CL:0002038', 'CL:0000788', 'CL:0000900', 'CL:0000670', 'CL:0002057', 'CL:0000351', 'CL:0001069', 'CL:0000091', 'CL:0000359', 'CL:0010004', 'CL:0000171', 'CL:0000169', 'CL:0000017', 'CL:0000623', 'CL:0001057', 'CL:0002394', 'CL:0000129']
    ref_group = ['full','500', '001', '005', '01', '02', '03', '05', '07', '09']
    dirichlet_para = [0.001, 0.01, 0.1, 1]

    for i in ref_group:
        signature = np.loadtxt('/ua/shi235/IndependentStudy/Data/Signature/signature_' + i + "_" + str(exp_index) + ".tsv", delimiter="\t")
        for j in dirichlet_para:
            data = np.loadtxt('/ua/shi235/IndependentStudy/Data/deconvoluteData/data_' + str(j) +  "_" +i + "_" + str(exp_index) + ".tsv", delimiter="\t")
            
            # Mixture
            Gene_symbol = np.arange(0,data.reshape((1,len(data))).shape[1])
            a = pd.DataFrame(data, index=Gene_symbol)
            a.to_csv('/ua/shi235/IndependentStudy/Data/tempdata/mixture_cibersort.txt', sep='\t')

            # Signature
            signature = pd.DataFrame(signature, index=Gene_symbol, columns= qualified_cell_type)
            # signature_dtangle = signature_dtangle.transpose()
            signature.to_csv('/ua/shi235/IndependentStudy/Data/tempdata/signature_cibersort.txt', sep='\t')

            # Run
            os.system('docker run -v /ua/shi235/IndependentStudy/Data/tempdata:/src/data -v /ua/shi235/IndependentStudy/Data/tempdata:/src/outdir cibersortx/fractions & docker run -v /ua/shi235/IndependentStudy/Data/tempdata:/src/data -v /ua/shi235/IndependentStudy/Data/tempdata:/src/outdir cibersortx/fractions --username shi235@wisc.edu --token 37c9348786eba1c59f67fe4cc1106ac9 --sigmatrix /ua/shi235/IndependentStudy/Data/tempdata/signature_cibersort.txt --mixture /ua/shi235/IndependentStudy/Data/tempdata/mixture_cibersort.txt --QN FALSE')

            # Read Data
            result = pd.read_csv('ua/shi235/IndependentStudy/Data/tempdata/CIBERSORTx_Results.txt', sep = '\t', index_col=0).iloc[:,0:48].to_numpy()
            # print(result)
            np.savetxt("/ua/shi235/IndependentStudy/Data/EstimatedResult/cibersort/proportion_" + str(exp_index) + "_" + str(j) + "_" + i + ".tsv", result)
            break
        break

if __name__ == "__main__":

    # model = SP1_model.build_model_cv_mean()
    
    # model = SP1_model.cell_type_variance()
    # model = SP1_model.cell_type_variance_model()
    model = 1

    print("finished learning model")
    # Parallel(n_jobs=50)(
    #     delayed(nnls)(exp) for exp in range(100)
    # )
    
    Parallel(n_jobs=50)(
        delayed(weightedIRLS)(exp, model) for exp in range(100)
    )

    # Parallel(n_jobs=30)(
    #     delayed(dtangle)(exp) for exp in range(100)
    # )

    # Parallel(n_jobs=1)(
    #     delayed(fardeep)(exp) for exp in range(100)
    # )

    # Parallel(n_jobs=1)(
    #     delayed(cibersort)(exp) for exp in range(1)
    # )

    # Parallel(n_jobs=50)(
    #     delayed(dwls)(exp) for exp in range(100)
    # )
