import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('WebAgg')

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
# import pymc

filename = 'PE_analysis_more_samps'

new_fit = False

data = hddm.load_csv('%s.csv'%filename)


# fit initial model
m = hddm.HDDM(data[data['missed_response']==0], depends_on = {'v':['TR_PE','TI_PE'],'t':['TR_PE','TI_PE']})


embed()


# m = hddm.HDDM(data[data['missed_response']==0], depends_on = {'v':['TR_PE','TI_PE']})

# m.find_starting_values()

# runii = 1

# m.sample(500, burn = 250, dbname = '%s-%i-db.pickle'%(filename,runii), db = 'pickle')


# embed()
