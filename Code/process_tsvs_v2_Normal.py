import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import pipeline2 as v5
import cell_ontology as co

file_name = 'expression_CPM.h5'
decon_temp = './decon_temp/'
decon_temp_shell = "./decon_temp/"
single_exp_list = ["SRP109837",
"SRP058654",
"SRP055180",
"SRP022259",
"SRP041597",
"SRP072398",
"SRP009105",
"SRP014320",
"SRP075318",
"SRP015909",
"SRP056074",
"SRP092128",
"SRP080962",
"SRP056146",
"SRP007338",
"SRP057826",
"SRP103769",
"SRP064121",
"SRP049028",
"SRP014620",
"SRP111077",
"SRP009029",
"SRP082575",
"SRP065317",
"SRP057616",
"SRP095330",
"SRP118609",
"SRP001371",
"SRP002105",
"SRP058500"]

def get_proportion(exp, nonzero, singleExp):
    labels = v5.exp_to_celltypes(exp)
    celltypes = []

    for label in labels:
        celltypes.append(co.get_term_name(label))
        for descendant in co.get_descendents(label):
            celltypes.append(co.get_term_name(descendant))
#     print(nonzero.columns)
    expright = 0
    ll = 0
#     print(nonzero)
    for i in range(ll, len(nonzero['0'])):
        if nonzero['0'][i] == exp and nonzero['1'][i] in celltypes:
            expright += nonzero['2'][i]
            ll = i
    
    if singleExp:
        expright = expright/2
#     if expright > 1:
#         expright = expright/2

    return expright

def get_exp_proportion(exp):
    study = v5.exp_to_study(exp)
    nonzero = pd.read_csv(decon_temp + 'nonzero_Normal_'
            + study + '.tsv', sep = '\t',index_col=0)
    
    exps = v5.study_to_exps(study)
    if len(exps) == 1:
        singleExp = True
    else:
        singleExp = False

    return get_proportion(exp, nonzero,singleExp)

def get_study_proportion(study):
    nonzero = pd.read_csv(decon_temp + 'nonzero_Normal_'
            + study + '.tsv', sep = '\t',index_col=0)
    exps = v5.study_to_exps(study)
    
    if study not in single_exp_list:
        singleExp = False
    else:
        singleExp = True

    proportion = []
    for exp in exps:
        proportion.append(get_proportion(exp, nonzero, singleExp))
    avg = np.mean(np.array(proportion))

    return avg

def celltype_to_proportion(type_list, descendants = False):
    explist = []
    for celltype in type_list:
        if descendants:
            explist += v5.ancestor_exps(celltype)
        else:
            explist = v5.celltype_to_exp(celltype)
    proportion = []
    for exp in explist:
#         try:
#             proportion.append(get_exp_proportion(exp))
#         except:
#             print(exp)
#             pass
        proportion.append(get_exp_proportion(exp))

    return np.array(proportion)

def proportions_all():
    """
    Returns a dictionary with experiment accession numbers as keys and
    proportions correct as values.
    """
    # update because that makes no sense
    proportions = {}
    studies = v5.get_studies()
    for study in studies:
#         try:
        nonzero = pd.read_csv(decon_temp + 'nonzero_Normal_'
                + study + '.tsv', sep = '\t')
        exps = v5.study_to_exps(study)
        
        if len(exps) == 1:
            singleExp = True
        else:
            singleExp = False
            
        for exp in exps:
            proportions[exp] = get_proportion(exp, nonzero, singleExp)
        
#         for exp in exps:
#             proportions[exp] = get_proportion(exp, nonzero)
#             if len(exps) == 1:
#                 print(study)
#                 continue
#             else:
#                 for exp in exps:
#                     proportions[exp] = get_proportion(exp, nonzero)
#         except:
#             print("Shit")
#             pass

    return proportions

def get_query_expression(exp):
    i = [v5.exp_to_index(exp)]
    query = v5.get_signatures(i)

    return query

def get_reference_expression(celltype, study, remove_study):
    # convert cell type to list of experiments
    exps = v5.celltype_to_exp(celltype)

    if remove_study:
        for exp in study:
            if exp in exps:
                exps.remove(exp)

    # convert experiment to indices
    indices = [v5.exp_to_index(exp) for exp in exps]

    # use index list to create reference
    reference = v5.get_signatures(indices)

    if len(indices) > 1:
        reference = np.mean(reference, axis = 1)

    return (reference, len(indices))

def scatterplot(ax, exp, celltype, remove_study = False):
    study = v5.exp_to_study(exp)
    if remove_study:
        if not is_type_available(celltype, study):
            print("Cell type ({}) not provided in reference ".format(celltype)
                    + "matrix for this study.")
            return

    query = get_query_expression(exp)
    exp_list = v5.study_to_exps(study)
    reference = get_reference_expression(celltype, exp_list, remove_study)

    ax.scatter(reference[0], query)
    ax.set(xlim=(0, 450000), ylim=(0, 450000))
    diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3") 
    xstick = np.arange(0, 450000 ,100000)
    ax.set_xticks(xstick)
    ax.set_yticks(xstick)
    ax.set(xlabel = "{} ({} experiments)".format(co.get_term_name(celltype),
            reference[1]), ylabel = exp)

def is_type_available(celltype, study):
    study = v5.study_to_exps(study)
    for exp in v5.get_exps():
        if exp in study:
            continue
        else:
            if celltype in v5.exp_to_celltypes(exp):
                return True
    return False

def study_type(study):
    ancestor_dict = {}
    exps = v5.study_to_exps(study)
    for exp in exps:
        ancestors = v5.exp_to_celltypes(exp)
        for celltype in v5.exp_to_celltypes(exp):
            ancestors += co.get_ancestors(celltype)
        ancestor_dict[exp] = ancestors

    if len(exps) == 1:
        ancestors = ancestor_dict[exps[0]]
    else:
        first = True
        for exp in ancestor_dict:
            if first:
                ancestors = set(ancestor_dict[exp])
                first = False
                continue
            ancestors = ancestors | set(ancestor_dict[exp])
        ancestors = list(ancestors)

    common = []
    for term_id in co.get_terms_without_children(list(ancestors)):
        common.append(co.get_term_name(term_id))
    common = '; '.join(common)

    return common

def exp_to_celltypes(exp):
    ids = v5.exp_to_celltypes(exp)
    celltypes = [co.get_term_name(id) for id in ids]
    return '; '.join(celltypes)
