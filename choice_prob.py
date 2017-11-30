

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
# from numpy import *
# import scipy as sp
# from pandas import *

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

from math import *
import os,glob,sys,platform

import pickle as pickle
import pandas as pd

from IPython import embed

from BehaviorAnalyzer import BehaviorAnalyzer
from PupilAnalyzer import PupilAnalyzer
from Plotter import Plotter


from analysis_parameters import *

pl = Plotter(figure_folder = figfolder, linestylemap = linestylemap)



pupil_signals = {'correct': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)},
				 'incorrect': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)}}		 

pupil_signals_pile = {'correct': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)},
				 'incorrect': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)}}		 

reaction_times = {'correct': {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []},
				 'incorrect': {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}}		 




sublist_pile = {'correct': {'PP': [],
							 'UP': [],
							 'PU': [],
							 'UU': []},
				 'incorrect': {'PP': [],
							 'UP': [],
							 'PU': [],
							 'UU': []}
			 }		 



#		TASK:  		  COLOR	    ORI
condition_keymap = { 0: 'PP',  1: 'PP',
					10: 'PU', 20: 'PU',
					30: 'UP', 40: 'UP',
					50: 'UP', 60: 'UP'}

inverse_keymap = {'PP': [0,1],
				  'UP': [30,40,50,60],
				  'PU': [10,20]}#,
				  # 'UU': [50,60]}
embed()
for subname in sublist:

	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	# Initialize PA object
	pa = PupilAnalyzer(subname, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = stimulus_deconvolution_interval, verbosity = 0)


	# Combine signals based on condition	

	sub_signals = {'correct': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)},
								 'incorrect': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)}}		 

	sub_rts = {'correct': {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []},
					 'incorrect': {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}}	


	# Get trial-based, event-related, baseline-corrected signals centered on stimulus onset
	pa.signal_per_trial(only_correct = True, only_incorrect = False, reference_phase = 7, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = False, return_rt = True)

	for (key,signals) in pa.trial_signals.items():
		if len(signals)>0:
			sub_signals['correct'][condition_keymap[key]] = np.append(sub_signals['correct'][condition_keymap[key]], signals, axis=0)
			sub_rts['correct'][condition_keymap[key]].extend(pa.trial_rts[key])

	# Get trial-based, event-related, baseline-corrected signals centered on stimulus onset
	pa.signal_per_trial(only_correct = False, only_incorrect = True, reference_phase = 7, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = False, return_rt = True)

	for (key,signals) in pa.trial_signals.items():
		if len(signals)>0:
			sub_signals['incorrect'][condition_keymap[key]] = np.append(sub_signals['incorrect'][condition_keymap[key]], signals, axis=0)
			sub_rts['incorrect'][condition_keymap[key]].extend(pa.trial_rts[key])


	# for con in inverse_keymap.keys():
	# 	pupil_signals['correct'][con] = np.append(pupil_signals['correct'][con], np.mean(sub_signals['correct'][con], axis=0)[np.newaxis,:], axis=0)
	# 	pupil_signals['incorrect'][con] = np.append(pupil_signals['incorrect'][con], np.mean(sub_signals['incorrect'][con], axis=0)[np.newaxis,:], axis=0)

	for con in inverse_keymap.keys():
		pupil_signals['correct'][con] = np.append(pupil_signals['correct'][con], sub_signals['correct'][con].mean(axis=0)[np.newaxis,:], axis=0)
		pupil_signals['incorrect'][con] = np.append(pupil_signals['incorrect'][con], sub_signals['incorrect'][con].mean(axis=0)[np.newaxis,:], axis=0)	

		reaction_times['correct'][con].append(np.median(sub_rts['correct'][con]))
		reaction_times['incorrect'][con].append(np.median(sub_rts['incorrect'][con]))


