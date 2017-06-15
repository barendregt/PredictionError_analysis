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
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()

# from rpy2.robjects.packages import importr
# import rpy2.robjects as ro
# # ro.r('library("nlme", lib.loc="/usr/lib/R/library")')
# # import pandas.rpy.common as com
# stats = importr('stats')
# base = importr('base')



# nlme = importr('nlme')

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


from analysis_parameters import *

trial_deconvolution_interval = np.array([-1.5, 4.5])

pl = Plotter(figure_folder = figfolder)



all_correlations = {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []}


all_timepoint_data = [[]]*int((trial_deconvolution_interval[1]-trial_deconvolution_interval[0]) * deconv_sample_frequency)

# embed()

for subii,subname in enumerate(sublist):

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')


	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = trial_deconvolution_interval, verbosity = 0)

	pa.signal_per_trial(only_correct = True, only_incorrect = False, return_rt = True, reference_phase = 7, with_rt = True, baseline_type = 'relative', baseline_period = [-0.5, 0.0], down_sample = True)
	
	# trial_params = pa.read_trial_data(pa.combined_h5_filename)

	# key1 = condition
	# key2 = pe present/absent
	# key3 = tr-pe present/absent

	for key, trial_signal in pa.trial_signals.items():
		if key < 10:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==1) * (trial_params['trial_codes'] < 10)]
			key1 = 0
			key2 = 0
			key3 = 0
		elif key < 30:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==1) * (trial_params['trial_codes'] >= 10) * (trial_params['trial_codes'] < 30)]
			key1 = 1
			key2 = 1
			key3 = 1	
		elif key < 50:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==1) * (trial_params['trial_codes'] >= 30) * (trial_params['trial_codes'] < 50)]
			key1 = 2
			key2 = 1
			key3 = 0
		else:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==1) * (trial_params['trial_codes'] >= 50)]
			key1 = 3
			key2 = 1
			key3 = 1

		for timepoint in range(trial_signal.shape[1]):

			if len(all_timepoint_data[timepoint])==0:
				all_timepoint_data[timepoint] = np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*key3, np.ones((trial_signal.shape[0],1))*subii, np.ones((trial_signal.shape[0],1)), pa.trial_rts[key][:,np.newaxis]])
			else:
				all_timepoint_data[timepoint] = np.append(all_timepoint_data[timepoint], np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*key3, np.ones((trial_signal.shape[0],1))*subii, np.ones((trial_signal.shape[0],1)), pa.trial_rts[key][:,np.newaxis]]), axis=0)

	pa.signal_per_trial(only_correct = False, only_incorrect = True, return_rt = True, reference_phase = 7, with_rt = True, baseline_type = 'relative', baseline_period = [-0.5, 0.0], down_sample = True)

	for key, trial_signal in pa.trial_signals.items():
		if key < 10:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==0) * (trial_params['trial_codes'] < 10)]
			key1 = 0
			key2 = 0
			key3 = 0
		elif key < 30:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==0) * (trial_params['trial_codes'] >= 10) * (trial_params['trial_codes'] < 30)]
			key1 = 1
			key2 = 1
			key3 = 1	
		elif key < 50:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==0) * (trial_params['trial_codes'] >= 30) * (trial_params['trial_codes'] < 50)]
			key1 = 2
			key2 = 1
			key3 = 0
		else:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==0) * (trial_params['trial_codes'] >= 50)]
			key1 = 3
			key2 = 1
			key3 = 1

		if trial_signal.shape[0]>0:

			for timepoint in range(trial_signal.shape[1]):

				all_timepoint_data[timepoint] = np.append(all_timepoint_data[timepoint], np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*key3, np.ones((trial_signal.shape[0],1))*subii, np.zeros((trial_signal.shape[0],1)), pa.trial_rts[key][:,np.newaxis]]), axis=0)
		
all_timepoint_array = np.dstack(all_timepoint_data)



# main_PEtr = np.zeros((all_timepoint_array.shape[2],1))
# main_PEntr = np.zeros((all_timepoint_array.shape[2],1))
# main_correct = np.zeros((all_timepoint_array.shape[2],1))
# interact = np.zeros((all_timepoint_array.shape[2],1))



for timepoint in range(all_timepoint_array.shape[2]):
	testdata = pd.DataFrame(all_timepoint_array[:,:,timepoint], columns = ['pupil','condition','PE','TR','subject','correct','RT'], index=np.arange(all_timepoint_array.shape[0]))

	testdata.to_hdf('button_timepoints_matrix.h5','t%i'%timepoint, format = 't', data_columns = ['pupil','condition','PE','TR','subject','correct','RT'])

# 	mdl = stats.aov(ro.Formula('pupil ~ c(PEtr)*c(PEntr)*c(correct) + Error(subject)'), data=testdata)

# 	main_PEtr[timepoint] = float(base.summary(mdl)[1][0][4][0])
# 	main_PEntr[timepoint] = float(base.summary(mdl)[1][0][4][1])
# 	main_correct[timepoint] = float(base.summary(mdl)[1][0][4][2])
# 	interact[timepoint] = float(base.summary(mdl)[1][0][4][6])

all_timepoint_data = [[]]*int((trial_deconvolution_interval[1]-trial_deconvolution_interval[0]) * deconv_sample_frequency)

# embed()

for subii,subname in enumerate(sublist):

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')


	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = trial_deconvolution_interval, verbosity = 0)

	pa.signal_per_trial(only_correct = True, only_incorrect = False, return_rt = True, reference_phase = 4, with_rt = False, baseline_type = 'relative', baseline_period = [-0.5, 0.0], down_sample = True)
	
	# trial_params = pa.read_trial_data(pa.combined_h5_filename)

	# key1 = condition
	# key2 = pe present/absent
	# key3 = tr-pe present/absent

	for key, trial_signal in pa.trial_signals.items():
		if key < 10:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==1) * (trial_params['trial_codes'] < 10)]
			key1 = 0
			key2 = 0
			key3 = 0
		elif key < 30:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==1) * (trial_params['trial_codes'] >= 10) * (trial_params['trial_codes'] < 30)]
			key1 = 1
			key2 = 1
			key3 = 1	
		elif key < 50:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==1) * (trial_params['trial_codes'] >= 30) * (trial_params['trial_codes'] < 50)]
			key1 = 2
			key2 = 1
			key3 = 0
		else:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==1) * (trial_params['trial_codes'] >= 50)]
			key1 = 3
			key2 = 1
			key3 = 1

		for timepoint in range(trial_signal.shape[1]):

			if len(all_timepoint_data[timepoint])==0:
				all_timepoint_data[timepoint] = np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*key3, np.ones((trial_signal.shape[0],1))*subii, np.ones((trial_signal.shape[0],1)), pa.trial_rts[key][:,np.newaxis]])
			else:
				all_timepoint_data[timepoint] = np.append(all_timepoint_data[timepoint], np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*key3, np.ones((trial_signal.shape[0],1))*subii, np.ones((trial_signal.shape[0],1)), pa.trial_rts[key][:,np.newaxis]]), axis=0)

	pa.signal_per_trial(only_correct = False, only_incorrect = True, return_rt = True, reference_phase = 4, with_rt = False, baseline_type = 'relative', baseline_period = [-0.5, 0.0], down_sample = True)

	for key, trial_signal in pa.trial_signals.items():
		if key < 10:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==0) * (trial_params['trial_codes'] < 10)]
			key1 = 0
			key2 = 0
			key3 = 0
		elif key < 30:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==0) * (trial_params['trial_codes'] >= 10) * (trial_params['trial_codes'] < 30)]
			key1 = 1
			key2 = 1
			key3 = 1	
		elif key < 50:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==0) * (trial_params['trial_codes'] >= 30) * (trial_params['trial_codes'] < 50)]
			key1 = 2
			key2 = 1
			key3 = 0
		else:
			# tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==0) * (trial_params['trial_codes'] >= 50)]
			key1 = 3
			key2 = 1
			key3 = 1

		if trial_signal.shape[0]>0:

			for timepoint in range(trial_signal.shape[1]):

				all_timepoint_data[timepoint] = np.append(all_timepoint_data[timepoint], np.hstack([trial_signal[:,timepoint][:,np.newaxis], np.ones((trial_signal.shape[0],1))*key1, np.ones((trial_signal.shape[0],1))*key2, np.ones((trial_signal.shape[0],1))*key3, np.ones((trial_signal.shape[0],1))*subii, np.zeros((trial_signal.shape[0],1)), pa.trial_rts[key][:,np.newaxis]]), axis=0)
		
all_timepoint_array = np.dstack(all_timepoint_data)



# main_PEtr = np.zeros((all_timepoint_array.shape[2],1))
# main_PEntr = np.zeros((all_timepoint_array.shape[2],1))
# main_correct = np.zeros((all_timepoint_array.shape[2],1))
# interact = np.zeros((all_timepoint_array.shape[2],1))



for timepoint in range(all_timepoint_array.shape[2]):
	testdata = pd.DataFrame(all_timepoint_array[:,:,timepoint], columns = ['pupil','condition','PE','TR','subject','correct','RT'], index=np.arange(all_timepoint_array.shape[0]))

	testdata.to_hdf('stim_timepoints_matrix.h5','t%i'%timepoint, format = 't', data_columns = ['pupil','condition','PE','TR','subject','correct','RT'])