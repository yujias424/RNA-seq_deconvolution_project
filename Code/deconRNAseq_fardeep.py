import scipy.optimize.nnls as nnls
import numpy as np
from sklearn import preprocessing
from sklearn import decomposition
import pandas as pd
from pandas import DataFrame
import quadprog
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import math
from scipy.optimize import minimize 
from scipy.optimize import nnls 
import sklearn

def DeconNormal(signature_use, data_subdata, use_scale=False, L2_reg=False, L2_lamb=0.1):
    # number of cell/tissue types
    numofx = len(signature_use[0])

    # quadratic programming preparation
    signaturenp = signature_use
   
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

    BB = data_subdata
    if use_scale:
        BB = preprocessing.scale(BB)

    # Solve QP
    # Solve QP method
    
    # This step is required, without this, the result may generate non definite positive error
    if L2_reg == False:
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
    
    # NNLS approach (Better for handling L2 regularization)
    else:
        lamb = L2_lamb
        n_variables = AA.shape[1]
        # weight = np.eye(AA.shape[0])
        AA = np.concatenate([AA, np.sqrt(lamb)*np.eye(n_variables)])
        BB = np.concatenate([BB, np.zeros(n_variables)])
        x0, rnorm = nnls(AA,BB)

        # Define minimisation function
        def fn(x, A, b):
            return np.linalg.norm(A.dot(x) - b)

        # Define constraints and bounds
        cons = {'type': 'eq', 'fun': lambda x:  np.sum(x)-1}
        bounds = []
        for i in range(AA.shape[1]):
            bounds.append([0., None])

        #Call minimisation subject to these values
        minout = minimize(fn, x0, args=(AA, BB), method='SLSQP',bounds=bounds,constraints=cons)
        x = minout.x
        out = x
    
    out_all.append(out)
    out_all = np.array(out_all)

    out_all[out_all<tol] = 0.0 # Eliminate the near 0 count

    return out_all

def deconrnaseqweightIRLS(datasets, signatures, checksig = False, deconMethod = 'CVd',
            use_scale = False, addone = False, ProvideTM = None, true_std = None, 
                      drop_low = False, olrZero = False, drop_gene = False, threshold = 0, Proportion = None, L2_reg=False, L2_lamb=0.1):
    '''
    This is the weighted version DeconRNASeq, using the weighted matrix which is estimated using given dataset.
    Specifically, if wanted to use the estimated noise model, please provided the dataset containing several
    replicates, otherwise, the global version of weighted matrix will be applied.
    '''
    # First, we perform a normal version decon method to get an estimate of proportions.
    proportion = np.array(Proportion).reshape((1,len(signatures[0])))[0]
    avgexplist = np.matmul(signatures, proportion)

    # Weighted Processing
    # quadratic programming preparation
    signaturenp = signatures
    data_subdata = avgexplist
    numofx = len(signaturenp[0])

    # Calculate the Residual
    # residual = np.absolute(datasets-data_subdata)
    # residual_median = np.median(residual, axis=0)
    # large_residual_index = np.argwhere(residual>residual_median)
    # large_residual_index = [i[0] for i in large_residual_index]

    # print(large_residual_index)
    # print(len(datasets_111))

    # Step 2: Modify signaturenp and datanp
    tol = 2.220446e-16 # previous is 16
    
    deleteRow = []
    for i in range(len(avgexplist)):
        if avgexplist[i] == 0:
            deleteRow.append(i)
        else:
            if math.log(avgexplist[i]) > 6 or math.log(avgexplist[i]) <2:
                deleteRow.append(i)
    
    if drop_gene == True:
        signaturenp = np.delete(signaturenp, deleteRow, axis=0)
        data_subdata = np.delete(data_subdata, deleteRow, axis=0)
        avgexplist = np.delete(avgexplist, deleteRow, axis=0)
    
    # Build the weighted Matrix
    weightMatrix = buildweightmatrix(data_subdata, signaturenp, 
                                     avgexplist, varexplist=true_std,
                                     provideTM = ProvideTM,
                                     use_scale = use_scale,
                                     add_one = addone,
                                     weightedApproach = deconMethod, proportion = Proportion)

    replaceCount=0
    # Build the diagonal weighted matrix.
    
    histexp = [[],[],[],[],[],[]]
    if drop_low == False:
        for j in range(len(data_subdata)):
            if data_subdata[j] > threshold: 
                continue
            else:
                replaceCount += 1
                deleteRow.append(j)
    else:
        pass
                
    true_std = np.array(true_std)

    # Removed Gene
    # print(signaturenp.shape)
    # print(len(large_residual_index))
    # datasets = np.delete(datasets, large_residual_index)
    # signaturenp = np.delete(signaturenp, large_residual_index, axis = 0)
    # weightMatrix = np.delete(weightMatrix, large_residual_index)
    # print(signaturenp.shape)

    if use_scale:
        AA = preprocessing.scale(signaturenp)
    else:
        AA = signaturenp

    EE = np.array([1] * numofx)
    FF = np.array(1)
    HH = np.array([0] * numofx)
    GG = np.eye(numofx)

    out_all = []
        
#     BB = data_subdata
    BB = datasets

    if use_scale:
        BB = preprocessing.scale(BB)

    # Quadratic Programming Computation
    if L2_reg == False:
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

    # NNLS approach (Better for handling L2 regularization)
    else:
        lamb = L2_lamb
        n_variables = AA.shape[1]
        weight = np.diag(weightMatrix)
        # AA = np.matmul(weight,AA)
        # BB = np.matmul(weight,BB)
        AA = np.sqrt(weightMatrix)[:,None] * AA
        BB = np.sqrt(weightMatrix) * BB   
        AA = np.concatenate([AA, np.sqrt(lamb)*np.eye(n_variables)])
        BB = np.concatenate([BB, np.zeros(n_variables)])

        x0, rnorm = nnls(AA,BB)

        # Define minimisation function
        def fn(x, A, b):
            return np.linalg.norm(A.dot(x) - b)

        # Define constraints and bounds
        cons = {'type': 'eq', 'fun': lambda x:  np.sum(x)-1}
        bounds = []
        for i in range(AA.shape[1]):
            bounds.append([0., None])

        # Call minimisation subject to these values
        minout = minimize(fn, x0, args=(AA, BB), method='SLSQP',bounds=bounds,constraints=cons)
        x = minout.x
        out = x
        
    
    out_all.append(out)

    out_all = np.array(out_all)
    out_all[out_all<tol] = 0.0 # Eliminate the near 0 count
    

    return [out_all,weightMatrix,replaceCount,histexp]

def buildweightmatrix(data, signature, avgexplist, varexplist, 
                      use_scale = False, weightedApproach = "CVd", provideTM = None, add_one = False, proportion = None):
    '''
    data, signature: already loaded in deconWeight, in numpy array, and order is setted
                     up, that is data_subdata, signaturenp, avgexplist, varexplist
    Possible Approach:
    Cvd: model noise with parameter CV * datapoint
    datapoint
    '''
    
    avgexplist = np.average(signature, axis=1)
    weightedUse = []
    
    BB = data

    if use_scale:
        BB = preprocessing.scale(BB)

    # Transform data into log space
    tempBB = list(BB)
    for i in range(len(tempBB)):
        if tempBB[i] == 0:
            tempBB[i] = avgexplist[i]
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
        weightedUse = 1/np.array(varexplist+1)
        # print(weightedUse.shape)
    elif weightedApproach == 'std_prop':
        # print(proportion.shape)
        # print(varexplist.shape)
        weightedUse = 1/(np.matmul(np.square(proportion), varexplist)+1)
        # print(weightedUse.shape)
    elif weightedApproach == 'datapoint':
        tempBB += 1
        weightedUse = 1/tempBB
    elif weightedApproach == 'datapoint_cv':
        # print(tempBB.shape)
        # print(varexplist.shape)
        tmp_BB_exp = tempBB*varexplist
        # print(tmp_BB_exp.shape)
        weightedUse = 1/(tmp_BB_exp+1)
    elif weightedApproach == 'Estimate_std':
        if provideTM != None:
            tempBB = list(BB)
            for i in range(len(logBB)):
                estimate_log_std = provideTM(logBB[i])
                if estimate_log_std <= 700: # Overflow issue
                    tempBB[i] = math.exp(estimate_log_std)
                else:
                    tempBB[i] = math.exp(700)
#                 if tempBB[i] != 0:
#                     tempBB[i] = math.exp(provideTM(logBB[i]))
#                 else:
#                     tempBB[i] = math.exp(provideTM(logBB[i]))
            tempBB = np.array(tempBB)
            weightedUse = 1/np.square(tempBB)
        else:
            return "Please provide the required training model."
    
    return weightedUse

def deconrnaseqweight(datasets, signatures, checksig = False, deconMethod = 'CVd',
            use_scale = False, addone = False, ProvideTM = None, true_std = None, L2_lamb = 0.1,
                      drop_low = False, olrZero = False, drop_gene = False, threshold = 0, L2_reg=False):
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
    if drop_low == False:
        for j in range(len(data_subdata)):
            if data_subdata[j] > threshold: 
                continue
            else:
                continue
#                 replaceCount += 1
#                 deleteRow.append(j)
    else:
        pass

#     true_std = np.array(true_std)
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
    if L2_reg == False:
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

    # NNLS approach (Better for handling L2 regularization)
    else:
        lamb = L2_lamb
        n_variables = AA.shape[1]
        weight = np.diag(weightMatrix)
        AA = np.sqrt(weightMatrix)[:,None] * AA
        BB = np.sqrt(weightMatrix) * BB
        # AA = np.matmul(weight,AA)
        # BB = np.matmul(weight,BB)
        AA = np.concatenate([AA, np.sqrt(lamb)*np.eye(n_variables)])
        BB = np.concatenate([BB, np.zeros(n_variables)])

        x0, rnorm = nnls(AA,BB)

        # Define minimisation function
        def fn(x, A, b):
            return np.linalg.norm(A.dot(x) - b)

        # Define constraints and bounds
        cons = {'type': 'eq', 'fun': lambda x:  np.sum(x)-1}
        bounds = []
        for i in range(AA.shape[1]):
            bounds.append([0., None])

        #Call minimisation subject to these values
        minout = minimize(fn, x0, args=(AA, BB), method='SLSQP',bounds=bounds,constraints=cons)
        x = minout.x
        out = x
    
    out_all.append(out)

    out_all = np.array(out_all)
    out_all[out_all<tol] = 0.0 # Eliminate the near 0 count

    return [out_all,weightMatrix,replaceCount,histexp]

def variance_matrix(observed_data, signature, estimated_proportion):
    # This function is to calculate the new variance matrix based on the provided data set.

    new_data = np.matmul(signature, estimated_proportion)
    err = np.absolute(observed_data - new_data)
    
