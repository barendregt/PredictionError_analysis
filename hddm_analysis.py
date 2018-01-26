import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('qt')

# import matplotlib.pyplot as plt
# import seaborn as sn

# from joblib import Parallel, delayed
# import multiprocessing

from math import *
import os,glob,sys,platform

import pickle
import pandas as pd

from IPython import embed

import hddm
import kabuki
# import pymc

filename = 'PE_analysis_more_samps'

new_fit = False

data = hddm.load_csv('%s.csv'%filename)


use_data = data[(data['rt'] > 0) & (data['rt'] < 4)]

use_data = hddm.utils.flip_errors(use_data)

# fit initial model






# m = hddm.HDDM(data[data['missed_response']==0], depends_on = {'v':['TR_PE','TI_PE']})



models = []

for runii in range(5):

	m = hddm.load('model-run%i'%runii)

	models.append(m)



embed()

# # def run_hddm(runii, use_data):

# 	m = hddm.HDDM(use_data, p_outlier = 0.05, depends_on = {'v':['TR_PE','TI_PE'],'t':['TR_PE','TI_PE']})

# 	m.find_starting_values()

# 	m.sample(5000, burn = 2500, dbname = '%s-%i-db.pickle'%(filename,runii), db = 'pickle')

# 	m.save('model-run%i'%runii)

	# models.append(m)

# Run everything in parallel for speed 
# num_jobs = 5

# Parallel(n_jobs=num_jobs)(delayed(run_hddm)(runii, use_data) for runii in range(num_jobs))	

# hddm.analyze.gelman_rubin(models)
# embed()
