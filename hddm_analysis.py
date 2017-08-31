import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

from math import *
import os,glob,sys,platform

import pickle
import pandas as pd

from IPython import embed

import hddm


filename = 'for_hddm.csv'

data = hddm.load_csv(filename)


# fit initial model
m = hddm.HDDM(data[data['missed_response']>0], depends_on = {'v':['TR_PE','TI_PE'],'t':['TR_PE','TI_PE']})

#m = hddm.HDDM(data[data['missed_response']>0], depends_on = {'v':['TR_PE','TI_PE']})

m.find_starting_values()

m.sample(100, burn = 50, dbname = 'dbresults', db = 'pickle')

embed()