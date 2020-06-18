import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
import pandas as pd
from pandas import DataFrame
from scipy.optimize import lsq_linear, nnls, minimize
import quadprog
from sklearn.metrics import mean_squared_error
from math import sqrt
from time import time
import scipy
import random
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
import math

"""
Normal DeconRNASeq
"""
def deconrnaseq(datasets, signatures, proportions = None, checksig=False,
               known_prop=False, use_scale= True, fig=True):

    """
    Normal edition deconrnaseq method, use nnls without weighting the signature matrix.
    """

    # Check input requirement
    if datasets == None:
        return " Missing the mixture dataset, please provide a tab-delimited text file for mixture samples."
    if signatures == None: 
        return " Missing the signature dataset, please provide a tab-delimited text file for pure tissue/cell types."
    if proportions == None and known_prop: 
        return " Missing the known proprotions, please provide a tab-delimited text file containing known fractions for pure tissue/cell types."

    # Load Data
    try:
        signature = pd.read_csv(signatures, index_col = 0, sep='\t')
        data = pd.read_csv(datasets, index_col = 0, sep='\t')
    except:
        return " Invalid file is provided. "

    # To np array
    signaturenp = signature.to_numpy()
    datanp = data.to_numpy()

    # Get the column name
    columnName = signature.columns.values

    # Get the row name
    rowName = data.columns.values

    # Check error
    if isinstance(signaturenp, np.ndarray) == False:
        return "signature datasets must be a dataframe"
    if np.isnan(signaturenp).any():
        return "signature data cannot have NAs. please exclude or impute missing values."
    if isinstance(datanp, np.ndarray) == False:
        return "mixture datasets must be a dataframe"
    if np.isnan(datanp).any():
        return "mixture data cannot have NAs. please exclude or impute missing values."

    signatureShape = np.shape(signaturenp)
    numofg = signatureShape[0] # Number of the row of siganture matrix
    numofx = signatureShape[1] # Number of the column of signature matrix

    if numofg < numofx:
        return "The number of genes is less than the number of cell types, which means less independent equations than unknowns."

    ## reorder the rows to match the signature file
    common_signature = signature.index.isin(data.index)
    common_data = data.index.isin(signature.index)
    signature_use = signature[common_signature == True]
    data_subdata = data[common_data == True]

    # number of cell/tissue types
    numofx = len(signature_use.columns)

    # quadratic programming preparation
    signaturenp = signature_use.to_numpy()
    if use_scale:
        AA = preprocessing.scale(signaturenp)
    else:
        AA = signaturenp

    EE = np.array([1] * numofx)
    FF = np.array(1)
    HH = np.array([0] * numofx)
    GG = np.eye(numofx)

    out_all = []
    tol = 2.220446e-16
    for i in range(len(data_subdata.columns)):

        BB = data_subdata.iloc[:,i].to_numpy()
        if use_scale:
            BB = preprocessing.scale(BB)

        # Solve QP
        # Solve QP method
        dvec = np.matmul(np.transpose(AA),BB.reshape(len(BB),1))
        dvec = dvec.reshape(dvec.shape[0])
        Dmat = np.matmul(np.transpose(AA), AA)

        temp = [1e-08] * len(Dmat)
        Dmat = Dmat + np.diag(temp)

        Amat = np.concatenate((EE.reshape(len(EE),1), GG),axis=1)
        bvec = np.concatenate((FF, HH), axis=None)
        bvec=bvec.astype('double')
        Amat=Amat.astype('double')

        sc = np.linalg.norm(Dmat)
        out = quadprog.solve_qp(Dmat/sc ,dvec/sc, Amat, bvec, meq=1)[0]
        out_all.append(out)
    out_all = np.array(out_all)

    out_all[out_all<tol] = 0.0 # Eliminate the near 0 count

    df = DataFrame(out_all, columns = columnName, index = rowName)

    mean_rmse = 0

    result = [df, mean_rmse]

    return result

"""
IRLS deconrnaseq
"""
def deconrnaseqweightCore(datasets, signatures, proportions = None, checksig=False,
               known_prop=False, use_scale= False, fig=True, Trainmodel_1 = None, Trainmodel_2 = None):
    '''
    This is the weighted version DeconRNASeq, using the weighted matrix which is estimated using given dataset.
    Specifically, if wanted to use the estimated noise model, please provided the dataset containing several
    replicates, otherwise, the global version of weighted matrix will be applied.
    '''

    # Check input requirement
    if datasets == None:
        return " Missing the mixture dataset, please provide a tab-delimited text file for mixture samples."
    if signatures == None: 
        return " Missing the signature dataset, please provide a tab-delimited text file for pure tissue/cell types."
    if proportions == None and known_prop: 
        return " Missing the known proprotions, please provide a tab-delimited text file containing known fractions for pure tissue/cell types."

    # Load Data
    signature = pd.read_csv(signatures, index_col = 0, sep='\t')
    data = pd.read_csv(datasets, index_col = 0, sep='\t')

    # To np array
    signaturenp = signature.to_numpy()
    datanp = data.to_numpy()

    # Get the column name
    columnName = signature.columns.values

    # Get the row name
    rowName = data.columns.values

    # Check error
    if isinstance(signaturenp, np.ndarray) == False:
        return "signature datasets must be a dataframe"
    if np.isnan(signaturenp).any():
        return "signature data cannot have NAs. please exclude or impute missing values."
    if isinstance(datanp, np.ndarray) == False:
        return "mixture datasets must be a dataframe"
    if np.isnan(datanp).any():
        return "mixture data cannot have NAs. please exclude or impute missing values."

    signatureShape = np.shape(signaturenp)
    numofg = signatureShape[0] # Number of the row of siganture matrix
    numofx = signatureShape[1] # Number of the column of signature matrix

    if numofg < numofx:
        return "The number of genes is less than the number of cell types, which means less independent equations than unknowns."

    ## reorder the rows to match the signature file
    common_signature = signature.index.isin(data.index)
    common_data = data.index.isin(signature.index)
    signature_use = signature[common_signature == True]
    data_subdata = data[common_data == True]

    # number of cell/tissue types
    numofx = len(signature_use.columns)

    # Weighted Processing
    # Step 1: Create average expression level across all cell type
    avgexplist = np.average(signature_use.to_numpy(), axis=1)
    varexplist = np.std(signature_use.to_numpy(), axis=1)

    # quadratic programming preparation
    signaturenp = signature_use.to_numpy()
    data_subdata = data_subdata.to_numpy()

    # Step 2: Modify signaturenp and datanp
    tol = 2.220446e-16 # previous is 16
    nrow = signaturenp.shape[0]
    ncol = signaturenp.shape[1]

    out_all = []

    k = 0
    for i in range(len(data_subdata[1])):
        
        out = deconrnaseqweight(data_subdata[:,i], signaturenp, deconMethod = 'Estimate_std',
            use_scale = False, addone = True, ProvideTM = Trainmodel_2, true_std = None, 
                      drop_low = True, olrZero = False, drop_gene = False)[0]
        
        for j in range(5):
            out_temp = out
            out = deconrnaseqweightIRLS(data_subdata[:,i], signaturenp, deconMethod = 'Estimate_std',
            use_scale = False, addone = True, ProvideTM = Trainmodel_1, true_std = None, 
                      drop_low = True, olrZero = False, drop_gene = False, Proportion=out_temp)[0]
            
        # break
        out_all.append(out.tolist())
    # print(out_all)
    out_all = np.array(out_all)
    out_all[out_all<tol] = 0.0 # Eliminate the near 0 count
    # print(out_all.shape)
    df = DataFrame(out_all, columns = columnName, index = rowName)

    # Draw rmse graph
    mean_rmse = 0

    result = [df, mean_rmse]

    return result

"""
Build weighted matrix in IRLS first step
"""
def buildweightmatrix(data, signature, avgexplist, varexplist, 
                      use_scale = False, weightedApproach = "CVd", provideTM = None, add_one = False):
    '''
    data, signature: already loaded in deconWeight, in numpy array, and order is setted
                     up, that is data_subdata, signaturenp, avgexplist, varexplist
    Possible Approach:
    Cvd: model noise with parameter CV * datapoint
    datapoint
    '''
    
    avgexplist_signature = np.average(signature, axis=1)
    weightedUse = []
    
    BB = data

    if use_scale:
        BB = preprocessing.scale(BB)

    # Transform data into log space
    tempBB = list(BB)
    for i in range(len(tempBB)):
        if tempBB[i] == 0:
            tempBB[i] = avgexplist_signature[i]
    tempBB = np.array(tempBB)
    if add_one == True:
        logBB = np.log(tempBB+1) #Need +1 Here?
    else:
        logBB = np.log(tempBB)
    
    # Applied the given training model
    if weightedApproach == 'CVd':
        if provideTM != None:
            tempBB = list(BB)
            for i in range(len(logBB)):
                if tempBB[i] != 0:
                    tempBB[i] = math.exp(provideTM(logBB[i]))*tempBB[i]
                else:
                    tempBB[i] = math.exp(provideTM(logBB[i]))*avgexplist[i]
            tempBB = np.array(tempBB)
            weightedUse = 1/tempBB
        else:
            return "Please provide the required training model."
    elif weightedApproach == 'CV':
        if provideTM != None:
            tempBB = list(BB)
            for i in range(len(logBB)):
                if tempBB[i] != 0:
                    tempBB[i] = math.exp(provideTM(logBB[i]))#*tempBB[i]
                else:
                    tempBB[i] = math.exp(provideTM(logBB[i]))#*avgexplist[i]
            tempBB = np.array(tempBB)
            weightedUse = 1/tempBB
        else:
            return "Please provide the required training model."
    elif weightedApproach == 'normal':
        weightedUse = [1] * len(list(BB))
        weightedUse = np.array(weightedUse)
    elif weightedApproach == 'std':
        weightedUse = 1/np.array(varexplist)
    elif weightedApproach == 'datapoint':
        weightedUse = 1/tempBB
    elif weightedApproach == 'Estimate_std':
        if provideTM != None:
            tempBB = list(BB)
            for i in range(len(logBB)):
                estimate_log_std = provideTM(logBB[i])
                if estimate_log_std <= 700: # Overflow issue
                    tempBB[i] = math.exp(estimate_log_std)
                else:
                    tempBB[i] = math.exp(700)
            tempBB = np.array(tempBB)
            weightedUse = 1/tempBB
        else:
            return "Please provide the required training model."
        
    return weightedUse

"""
Build Weighted Matrix for IRLS deconrnaseq
"""
def buildweightmatrix1(data, signature, avgexplist, varexplist, 
                      use_scale = False, weightedApproach = "CVd", provideTM = None, add_one = False):
    '''
    data, signature: already loaded in deconWeight, in numpy array, and order is setted
                     up, that is data_subdata, signaturenp, avgexplist, varexplist
    Possible Approach:
    Cvd: model noise with parameter CV * datapoint
    datapoint
    '''
    
    weightedUse = []
    
    BB = data

    if use_scale:
        BB = preprocessing.scale(BB)

    # Transform data into log space
    tempBB = list(BB)
    for i in range(len(tempBB)):
        tempBB[i] = avgexplist[i]
    tempBB = np.array(tempBB)
    if add_one == True:
        logBB = np.log(tempBB+1)
    else:
        logBB = np.log(tempBB)
    
    # Applied the given training model
    if weightedApproach == 'CVd':
        if provideTM != None:
            tempBB = list(BB)
            for i in range(len(logBB)):
                if tempBB[i] != 0:
                    tempBB[i] = math.exp(provideTM(logBB[i]))*tempBB[i]
                else:
                    tempBB[i] = math.exp(provideTM(logBB[i]))*avgexplist[i]
            tempBB = np.array(tempBB)
            weightedUse = 1/tempBB
        else:
            return "Please provide the required training model."
    elif weightedApproach == 'CV':
        if provideTM != None:
            tempBB = list(BB)
            for i in range(len(logBB)):
                if tempBB[i] != 0:
                    tempBB[i] = math.exp(provideTM(logBB[i]))#*tempBB[i]
                else:
                    tempBB[i] = math.exp(provideTM(logBB[i]))#*avgexplist[i]
            tempBB = np.array(tempBB)
            weightedUse = 1/tempBB
        else:
            return "Please provide the required training model."
    elif weightedApproach == 'normal':
        weightedUse = [1] * len(list(BB))
        weightedUse = np.array(weightedUse)
    elif weightedApproach == 'std':
        weightedUse = 1/np.array(varexplist)
    elif weightedApproach == 'datapoint':
        weightedUse = 1/tempBB
    elif weightedApproach == 'Estimate_std':
        if provideTM != None:
            tempBB = list(BB)
            for i in range(len(logBB)):
                # Overall, it is impossible that in real dataset, some expression level be that extremely high
                # Therefore, it is OK to use math.exp(provideTM(logBB[i]))
                tempBB[i] = math.exp(provideTM(logBB[i]))
            tempBB = np.array(tempBB)
            weightedUse = 1/tempBB
        else:
            return "Please provide the required training model."

    return weightedUse

def deconrnaseqweight(datasets, signatures, checksig = False, deconMethod = 'CVd',
            use_scale = False, addone = False, ProvideTM = None, true_std = None, 
                      drop_low = False, olrZero = False, drop_gene = False):
    '''
    This is the weighted version DeconRNASeq, using the weighted matrix which is estimated using given dataset.
    Specifically, if wanted to use the estimated noise model, please provided the dataset containing several
    replicates, otherwise, the global version of weighted matrix will be applied.
    '''

    # Load Data
    signature = signatures #pd.read_csv(signatures, index_col = 0, sep='\t')
    data = datasets #datasetspd.read_csv(datasets, index_col = 0, sep='\t')

    # To np array
    signaturenp = signature
    datanp = data
    signature_use = signaturenp
    data_subdata = datanp
    
    # number of cell/tissue types
    numofx = len(signature_use[0])

    # Weighted Processing
    # Step 1: Create average expression level across all cell type
    avgexplist = np.average(signature_use, axis=1)

    # quadratic programming preparation
    signaturenp = signature_use
    data_subdata = data_subdata

    # Step 2: Modify signaturenp and datanp
    tol = 2.220446e-16 # previous is 16
    
    # Build the weighted Matrix
    weightMatrix = buildweightmatrix(data_subdata, signaturenp, 
                                     avgexplist, varexplist=true_std,
                                     provideTM = ProvideTM,
                                     use_scale = use_scale,
                                     add_one = addone,
                                     weightedApproach = deconMethod)

    replaceCount=0
    # Build the diagonal weighted matrix.
    deleteRow = []
    histexp = [[],[],[],[],[],[]]
    # No need in real data analysis
    # if drop_low == False:
    #     pass
    # else:
    #     # Replace by true STD
    #     for j in range(len(data_subdata)):
    #         estimateSTD_Cal = np.exp(ProvideTM(np.log(data_subdata[j]+1)))
    #         if abs(estimateSTD_Cal - true_std[j])/estimateSTD_Cal > 1:
    #             if olrZero == True:
    #                 if data_subdata[j] == 0:
    #                     weightMatrix[j] = 1/true_std[j]
    #                     replaceCount += 1
    #                     deleteRow.append(j)
    #             else:
    #                 weightMatrix[j] = 1/true_std[j]
    #                 replaceCount += 1
    #                 deleteRow.append(j)
                
    #             histexp[0].append(data_subdata[j])
    #             histexp[1].append(estimateSTD_Cal)
    #             histexp[2].append(true_std[j])
                
    #         else:
    #             histexp[3].append(data_subdata[j])
    #             histexp[4].append(estimateSTD_Cal)
    #             histexp[5].append(true_std[j])

    true_std = np.array(true_std)
    if drop_gene == True:
        signaturenp = np.delete(signaturenp, deleteRow, axis=0)
        data_subdata = np.delete(data_subdata, deleteRow, axis=0)
        avgexplist = np.delete(avgexplist, deleteRow, axis=0)
        weightMatrix = np.array(weightMatrix)
        weightMatrix = np.delete(weightMatrix, deleteRow, axis=0)

    if use_scale:
        AA = preprocessing.scale(signaturenp)
    else:
        AA = signaturenp

    EE = np.array([1] * numofx)
    FF = np.array(1)
    HH = np.array([0] * numofx)
    GG = np.eye(numofx)

    out_all = []
        
    BB = data_subdata

    if use_scale:
        BB = preprocessing.scale(BB)
    
    # Quadratic Programming Computation
    dvec = np.multiply(np.transpose(AA),weightMatrix)
    dvec = np.matmul(dvec,BB.reshape(len(BB),1))
    dvec = dvec.reshape(dvec.shape[0])

    Dmat = np.multiply(np.transpose(AA), weightMatrix)
    Dmat = np.matmul(Dmat, AA)

    Amat = np.concatenate((EE.reshape(len(EE),1), GG),axis=1)
    bvec = np.concatenate((FF, HH), axis=None)

    bvec=bvec.astype('double')
    Amat=Amat.astype('double')

    sc = np.linalg.norm(Dmat)

    out = quadprog.solve_qp(Dmat/sc ,dvec/sc, Amat, bvec, meq=1)[0]
    out_all.append(out)

    out_all = np.array(out_all)
    out_all[out_all<tol] = 0.0 # Eliminate the near 0 count

    return [out_all,weightMatrix,replaceCount,histexp]

def deconrnaseqweightIRLS(datasets, signatures, checksig = False, deconMethod = 'CVd',
            use_scale = False, addone = False, ProvideTM = None, true_std = None, 
                      drop_low = False, olrZero = False, drop_gene = False, Proportion = None):
    '''
    This is the weighted version DeconRNASeq, using the weighted matrix which is estimated using given dataset.
    Specifically, if wanted to use the estimated noise model, please provided the dataset containing several
    replicates, otherwise, the global version of weighted matrix will be applied.
    '''
    # First, we perform a normal version decon method to get an estimate of proportions.
    proportion = np.array(Proportion).reshape((1,len(signatures[0])))[0]
    avgexplist = np.matmul(signatures, proportion)
    
    tol = 2.220446e-16
    deleteRow = []
    for i in range(len(avgexplist)):
        if avgexplist[i] <= 0:
            deleteRow.append(i)
            
    signaturenp = signatures
    data_subdata = datasets
    numofx = len(signaturenp[0])
    
    if drop_gene == True:
        signaturenp = np.delete(signaturenp, deleteRow, axis=0)
        data_subdata = np.delete(data_subdata, deleteRow, axis=0)
        avgexplist = np.delete(avgexplist, deleteRow, axis=0)
    
    # Build the weighted Matrix
    weightMatrix = buildweightmatrix1(avgexplist, signaturenp, 
                                     avgexplist, varexplist=true_std,
                                     provideTM = ProvideTM,
                                     use_scale = use_scale,
                                     add_one = addone,
                                     weightedApproach = deconMethod)

    replaceCount=0
    # Build the diagonal weighted matrix.
    deleteRow = []
    histexp = [[],[],[],[],[],[]]
    # if drop_low == False:
    #     for j in range(len(data_subdata)):
    #         if data_subdata[j] > threshold: 
    #             continue
    #         else:
    #             replaceCount += 1
    #             deleteRow.append(j)
    # else:
    #     pass
                
    true_std = np.array(true_std)

    if use_scale:
        AA = preprocessing.scale(signaturenp)
    else:
        AA = signaturenp

    EE = np.array([1] * numofx)
    FF = np.array(1)
    HH = np.array([0] * numofx)
    GG = np.eye(numofx)

    out_all = []
        
    BB = data_subdata

    if use_scale:
        BB = preprocessing.scale(BB)

    # Quadratic Programming Computation
    dvec = np.multiply(np.transpose(AA),weightMatrix)
    dvec = np.matmul(dvec,BB.reshape(len(BB),1))
    dvec = dvec.reshape(dvec.shape[0])

    Dmat = np.multiply(np.transpose(AA), weightMatrix)
    Dmat = np.matmul(Dmat, AA)

    Amat = np.concatenate((EE.reshape(len(EE),1), GG),axis=1)
    bvec = np.concatenate((FF, HH), axis=None)

    bvec=bvec.astype('double')
    Amat=Amat.astype('double')

    sc = np.linalg.norm(Dmat)

    out = quadprog.solve_qp(Dmat/sc ,dvec/sc, Amat, bvec, meq=1)[0]
    out_all.append(out)

    out_all = np.array(out_all)
    out_all[out_all<tol] = 0.0 # Eliminate the near 0 count
    

    return out_all