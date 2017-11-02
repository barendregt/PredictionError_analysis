

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from numpy import *
import scipy as sp
from pandas import *

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



pupil_signals = {'correct': {'PP': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float)},
				 'incorrect': {'PP': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float)}}		 


#		TASK:  		  COLOR	    ORI
condition_keymap = { 0: 'PP',  1: 'PP',
					10: 'PU', 20: 'PU',
					30: 'UP', 40: 'UP',
					50: 'UU', 60: 'UU'}

inverse_keymap = {'PP': [0,1],
				  'UP': [30,40],
				  'PU': [10,20],
				  'UU': [50,60]}

for subname in sublist:

	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	# Initialize PA object
	pa = PupilAnalyzer(subname, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = stimulus_deconvolution_interval, verbosity = 0)

	# Get trial-based, event-related, baseline-corrected signals centered on stimulus onset
	pa.signal_per_trial(only_correct = True, only_incorrect = False, reference_phase = 4, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = False)


	# Combine signals based on condition	

	sub_signals = {'correct': {'PP': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
								 'UP': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
								 'PU': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
								 'UU': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float)},
								 'incorrect': {'PP': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
								 'UP': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
								 'PU': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float),
								 'UU': np.empty((0,int((np.abs(stimulus_deconvolution_interval[0]) + np.abs(stimulus_deconvolution_interval[1]))*signal_sample_frequency)),dtype=float)}}		 


	for (key,signals) in pa.trial_signals.items():
		if len(signals)>0:
			sub_signals['correct'][condition_keymap[key]] = np.append(sub_signals['correct'][condition_keymap[key]], signals, axis=0)

	# Get trial-based, event-related, baseline-corrected signals centered on stimulus onset
	pa.signal_per_trial(only_correct = False, only_incorrect = True, reference_phase = 4, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = False)

	for (key,signals) in pa.trial_signals.items():
		if len(signals)>0:
			sub_signals['incorrect'][condition_keymap[key]] = np.append(sub_signals['incorrect'][condition_keymap[key]], signals, axis=0)


	# for con in inverse_keymap.keys():
	# 	pupil_signals['correct'][con] = np.append(pupil_signals['correct'][con], np.mean(sub_signals['correct'][con], axis=0)[np.newaxis,:], axis=0)
	# 	pupil_signals['incorrect'][con] = np.append(pupil_signals['incorrect'][con], np.mean(sub_signals['incorrect'][con], axis=0)[np.newaxis,:], axis=0)

	for con in inverse_keymap.keys():
		pupil_signals['correct'][con] = np.append(pupil_signals['correct'][con], sub_signals['correct'][con], axis=0)
		pupil_signals['incorrect'][con] = np.append(pupil_signals['incorrect'][con], sub_signals['incorrect'][con], axis=0)		


error_minus_noerror_correct = {'UP':[],
							   'PU':[],
							   'UU':[]}
error_minus_noerror_incorrect = {'UP':[],
							   'PU':[],
							   'UU':[]}
incorrect_minus_correct =  {'UP':[],
							   'PU':[],
							   'UU':[]}

for con in ['PU','UP','UU']:
	error_minus_noerror_correct[con] = pupil_signals['correct'][con]-pupil_signals['correct']['PP'].mean(axis=0)

for con in ['PU','UP','UU']:
	error_minus_noerror_incorrect[con] = pupil_signals['incorrect'][con]-pupil_signals['incorrect']['PP'].mean(axis=0)

for con in ['PU','UP','UU']:
	incorrect_minus_correct[con] = error_minus_noerror_incorrect[con].mean(axis=0) - error_minus_noerror_correct[con].mean(axis=0)

pl.open_figure(force=1)

pl.subplot(3,3,1)

pl.event_related_pupil_average(data=pupil_signals['correct'],x_lim=[25,250],xticks=np.arange(25,300,25),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.25,0.8],compute_mean=True, compute_sd = True, smooth_signal=True, smooth_factor=20, show_legend=True,title='Correct')

pl.subplot(3,3,3)

pl.event_related_pupil_average(data=pupil_signals['incorrect'],x_lim=[25,250],xticks=np.arange(25,300,25),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.25,0.8],compute_mean=True, compute_sd = True, smooth_signal=True, smooth_factor=20, show_legend=True,title='Incorrect')

pl.subplot(3,3,4)

pl.event_related_pupil_average(data=error_minus_noerror_correct,signal_labels={'UP':'UP-PP','PU':'PU-PP','UU':'UU-PP'},x_lim=[25,250],xticks=np.arange(25,300,25),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.1,0.25],compute_mean=True, compute_sd = True, smooth_signal=True, smooth_factor=20, show_legend=True)

pl.subplot(3,3,6)

pl.event_related_pupil_average(data=error_minus_noerror_incorrect,signal_labels={'UP':'UP-PP','PU':'PU-PP','UU':'UU-PP'},x_lim=[25,250],xticks=np.arange(25,300,25),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.1,0.25],compute_mean=True, compute_sd = True, smooth_signal=True, smooth_factor=20, show_legend=True)

pl.subplot(3,3,8)

pl.event_related_pupil_average(data=incorrect_minus_correct,x_lim=[25,250],xticks=np.arange(25,300,25),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.1,0.25],compute_mean=False, compute_sd = False, smooth_signal=True, smooth_factor=20, show_legend=True,title='Difference')

pl.save_figure(filename = 'pupil_differences.pdf', sub_folder = 'over_subs/pupil')

pl.close()

# plt.show()

# embed()