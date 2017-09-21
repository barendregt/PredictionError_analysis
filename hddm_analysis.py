import numpy as np
import scipy as sp

import matplotlib
# matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
import seaborn as sn

from math import *
import os,glob,sys,platform

import pickle
import pandas as pd

from IPython import embed

import hddm
# import pymc

filename = 'PE_analysis_upd'

new_fit = False

data = hddm.load_csv('%s.csv'%filename)


# fit initial model
m = hddm.HDDM(data[data['missed_response']==0], depends_on = {'v':['TR_PE','TI_PE'],'t':['TR_PE','TI_PE']})

# m = hddm.HDDM(data[data['missed_response']==0], depends_on = {'v':['TR_PE','TI_PE']})

m.find_starting_values()

m.sample(10000, burn = 5000, dbname = '%s-db.pickle'%filename, db = 'pickle')


embed()
# if new_fit:
# 	m.find_starting_values()

# 	m.sample(10, burn = 5, dbname = '%s-dbpick.pickle'%filename, db = 'pickle')

# 	m.save('%s-model'%filename)
# else:
# 	m.load_db('%s-db'%filename)

