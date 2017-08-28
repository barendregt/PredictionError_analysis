

import numpy as np
import scipy as sp

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from numpy import *
import scipy as sp
from pandas import *
# import readline
# from rpy2.robjects.packages import importr
# import rpy2.robjects as ro
# import pandas.rpy.common as com
# stats = importr('stats')
# base = importr('base')

import matplotlib.pyplot as plt 
import seaborn as sn 


sn.set(style='ticks', font='Arial', font_scale=1, rc={
	'axes.linewidth': 0.50, 
	'axes.labelsize': 7, 
	'axes.titlesize': 7, 
	'xtick.labelsize': 6, 
	'ytick.labelsize': 6, 
	'legend.fontsize': 6, 
	'xtick.major.width': 0.25, 
	'ytick.major.width': 0.25,
	'text.color': 'Black',
	'axes.labelcolor':'Black',
	'xtick.color':'Black',
	'ytick.color':'Black',} )

from math import *
import os,glob,sys,platform

import pickle as pickle
import pandas as pd

from IPython import embed

# sys.path.append('tools/')
from BehaviorAnalyzer import BehaviorAnalyzer
from Plotter import Plotter


if platform.node()=="aeneas":
	raw_data_folder = '/home/raw_data/2017/visual/PredictionError/Behavioural/Reaction_times/'
else:
	raw_data_folder = '/home/barendregt/Projects/PredictionError/Psychophysics/Data/k1f46/' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/PredictionError/Figures'

sublist = ['AB','AC','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX']#,'BA','BB','BC','BD','BE','BF','BG','BH']#
# sublist = ['AA','AB','AC','AD','AF','AG','AH','AI','AJ','AM']
# sublist_pos = ['AA','AB','AG','AJ','AL','AM','AO','AC','AF','AH','AI','AK','AN','AO','AP']
# sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AD','AE','AK','AL','AM','AN']
sbsetting = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
trial_deconvolution_interval = np.array([-1, 1.5])
# trial_deconvolution_interval = np.array([-1, 3])

down_fs = int(signal_sample_frequency / deconv_sample_frequency)

# linestylemap = {'PP': ['k-'],
# 				 'UP': [''],
# 				 'PU': [''],
# 				 'UU': ['']}

pl = Plotter(figure_folder = figfolder)



all_tprs = {'PP': [],
			 'UP': [],
			 'PU': [],
			 'UU': []}

all_responses = {'PP': [],
			 'UP': [],
			 'PU': [],
			 'UU': []}

all_directions = {'PP': [],
			 'UP': [],
			 'PU': [],
			 'UU': []}

all_data = [[]] * 20000
all_data_index = 0

for subname in sublist:

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	#pl.figure_folder = os.path.join(rawfolder,'results/')

	# if not os.path.isdir(os.path.join(rawfolder,'results/')):
	# 	os.makedirs(os.path.join(rawfolder,'results/'))

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = trial_deconvolution_interval, verbosity = 0)

	pa.compute_TPR(reference_phase = 7, time_window = [-1,1.5], baseline_period=[-0.5,0.0], with_rt=True)

	trial_parameters = pa.read_trial_data(pa.combined_h5_filename)

	for key, tpr in list(pa.TPR.items()):
		if key < 10:

			all_data[all_data_index] = np.hstack([tpr, trial_parameters['response'][trial_parameters['trial_codes']==key], trial_parameters['trial_direction'][trial_parameters['trial_codes']==key], 'PP'])

			all_tprs['PP'].extend(tpr)
			all_responses['PP'].extend(trial_parameters['response'][trial_parameters['trial_codes']==key])
			all_directions['PP'].extend(trial_parameters['trial_direction'][trial_parameters['trial_codes']==key])
		elif key < 30:
			all_tprs['UP'].extend(tpr)
			all_responses['UP'].extend(trial_parameters['response'][trial_parameters['trial_codes']==key])
			all_directions['UP'].extend(trial_parameters['trial_direction'][trial_parameters['trial_codes']==key])			
		elif key < 50:
			all_tprs['PU'].extend(tpr)
			all_responses['PU'].extend(trial_parameters['response'][trial_parameters['trial_codes']==key])
			all_directions['PU'].extend(trial_parameters['trial_direction'][trial_parameters['trial_codes']==key])			
		else:
			all_tprs['UU'].extend(tpr)
			all_responses['UU'].extend(trial_parameters['response'][trial_parameters['trial_codes']==key])
			all_directions['UU'].extend(trial_parameters['trial_direction'][trial_parameters['trial_codes']==key])			

	



	embed()

