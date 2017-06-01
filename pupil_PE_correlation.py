from __future__ import division

import numpy as np
import scipy as sp

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from numpy import *
import scipy as sp
from pandas import *

# R stuff
# import readline
from rpy2.robjects import pandas2ri
pandas2ri.activate()

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
# import pandas.rpy.common as com
stats = importr('stats')
base = importr('base')

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

import cPickle as pickle
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

sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV']#,'BA','BB','BC','BD','BE','BF','BG','BH']#
# sublist = ['AA','AB','AC','AD','AF','AG','AH','AI','AJ','AM']
# sublist_pos = ['AA','AB','AG','AJ','AL','AM','AO','AC','AF','AH','AI','AK','AN','AO','AP']
# sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AD','AE','AK','AL','AM','AN']
sbsetting = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
trial_deconvolution_interval = np.array([-1.5, 4.5])
# trial_deconvolution_interval = np.array([-1, 3])

down_fs = int(signal_sample_frequency / deconv_sample_frequency)


pl = Plotter(figure_folder = figfolder)



all_correlations = {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []}


all_timepoint_data = [[]]*int((trial_deconvolution_interval[1]-trial_deconvolution_interval[0]) * signal_sample_frequency)



for subii,subname in enumerate(sublist):

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')


	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = trial_deconvolution_interval, verbosity = 0)

	pa.signal_per_trial(only_correct = True, only_incorrect = False, reference_phase = 7, with_rt = True, baseline_type = 'relative', baseline_period = [-0.5, 0.0])

	for key, trial_signal in pa.trial_signals.items():
		if key < 10:
			key1 = 0
			key2 = 0
		elif key < 30:
			key1 = 1
			key2 = 0
		elif key < 50:
			key1 = 0
			key2 = 1
		else:
			key1 = 1
			key2 = 1

		for timepoint in range(trial_signal.shape[1]):

			if len(all_timepoint_data[timepoint])==0:
				all_timepoint_data[timepoint] = np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*subii, np.ones((trial_signal.shape[0],1))])
			else:
				all_timepoint_data[timepoint] = np.append(all_timepoint_data[timepoint], np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*subii, np.ones((trial_signal.shape[0],1))]), axis=0)

	pa.signal_per_trial(only_correct = False, only_incorrect = True, reference_phase = 7, with_rt = True, baseline_type = 'relative', baseline_period = [-0.5, 0.0])

	for key, trial_signal in pa.trial_signals.items():
		if key < 10:
			key1 = 0
			key2 = 0
		elif key < 30:
			key1 = 1
			key2 = 0
		elif key < 50:
			key1 = 0
			key2 = 1
		else:
			key1 = 1
			key2 = 1

		if trial_signal.shape[0]>0:

			for timepoint in range(trial_signal.shape[1]):

				if len(all_timepoint_data[timepoint])==0:
					all_timepoint_data[timepoint] = np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*subii, np.zeros((trial_signal.shape[0],1))])
				else:
					all_timepoint_data[timepoint] = np.append(all_timepoint_data[timepoint], np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*subii, np.zeros((trial_signal.shape[0],1))]), axis=0)				
		
all_timepoint_array = np.dstack(all_timepoint_data)

testdata = pd.DataFrame(all_timepoint_data[0], columns = ['pupil','PEtr','PEntr','subject','correct'], index=np.arange(all_timepoint_data[0].shape[0]))

embed()