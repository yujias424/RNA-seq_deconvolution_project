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
import os

if __name__ == "__main__":
    # run_file = ['Analysis_marker_gene_finding_001.r','Analysis_marker_gene_finding_005.r','Analysis_marker_gene_finding_01.r','Analysis_marker_gene_finding_02.r','Analysis_marker_gene_finding_03.r','Analysis_marker_gene_finding_05.r','Analysis_marker_gene_finding_07.r','Analysis_marker_gene_finding_09.r']
    # run_file = ['Analysis_marker_gene_finding_091.r','Analysis_marker_gene_finding_093.r','Analysis_marker_gene_finding_095.r','Analysis_marker_gene_finding_097.r','Analysis_marker_gene_finding_099.r']
    
    run_file = ['Analysis_marker_gene_finding_01.r']
    
    # run_file = ['Analysis_marker_gene_finding_0995.r','Analysis_marker_gene_finding_0999.r','Analysis_marker_gene_finding_1.r']
    
    for i in run_file:
        runcode = '~/conda_env/py36/bin/Rscript ' + i
        os.system(runcode)