from __future__ import division

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

sn.set(style='ticks')

from math import *
import os,glob,sys

import cPickle as pickle
import pandas as pd

from IPython import embed

# sys.path.append('tools/')
from BehaviorAnalyzer import BehaviorAnalyzer
from Plotter import Plotter


raw_data_folder = '/home/raw_data/2017/visual/PredictionError/Behavioural/Reaction_times/'#/home/barendregt/Projects/PredictionError/Psychophysics/Data/k1f46/' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/PredictionError/Figures'

#sublist = ['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN']#
# sublist = ['AA','AB','AC','AD','AF','AG','AH','AI','AJ','AM']
sublist_pos = ['AA','AB','AG','AJ','AL','AM','AO']
sublist_neg = ['AC','AF','AH','AI','AK','AN']
# sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AD','AE','AK','AL','AM','AN']
sbsetting = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
trial_deconvolution_interval = np.array([-.5, 7])
stimulus_deconvolution_interval = np.array([-1, 3])

down_fs = 100

pl = Plotter(figure_folder = figfolder)



#### PLOT AVERAGES OVER SUBS

# pl.open_figure()

# pl.subplot(1,2,2, title= 'Average pupil difference')

power_time_window = [100,600]#[15,30]
zero_point = 15

# all_ie_scores = []
all_betas = []

response_fir_signals = {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []}

for subname in sublist_pos:

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

	recorded_pupil_signal = pa.read_pupil_data(pa.combined_h5_filename, signal_type='long_signal')

	resampled_pupil_signal = sp.signal.resample(recorded_pupil_signal, int((recorded_pupil_signal.shape[-1] / signal_sample_frequency)*deconv_sample_frequency), axis = -1)

	trial_params = pa.read_trial_data(pa.combined_h5_filename)

	tcodes = [0,10,30,50,70]

	pl.open_figure(force=1)
	pl.hline(y=0)

	tc_corr = [[]] * (len(tcodes)-1)
	for tcii in range(len(tcodes)-1):
		
		tc_rts = trial_params['reaction_time'][(trial_params['trial_codes'] >= tcodes[tcii]) * (trial_params['trial_codes'] < tcodes[tcii+1])]

		trials = trial_params['trial_phase_1_full_signal'][(trial_params['trial_codes'] >= tcodes[tcii]) * (trial_params['trial_codes'] < tcodes[tcii+1])] / signal_sample_frequency*deconv_sample_frequency

		tc_sigs = []
		for trii,trial in enumerate(trials):
			if (trial+(trial_deconvolution_interval[1]*signal_sample_frequency)) <= resampled_pupil_signal.shape[0]:
				tc_sigs.append(resampled_pupil_signal[trial+(trial_deconvolution_interval[0]*signal_sample_frequency):trial+(trial_deconvolution_interval[1]*signal_sample_frequency)])
			# else:
			# 	tc_rts = tc_rts.delete(trii)
		tc_sigs = np.vstack(tc_sigs)
		tc_rts = tc_rts[:tc_sigs.shape[0]]
		tc_corr = []
		for timep in range(tc_sigs.shape[1]):
			tc_corr.append(np.corrcoef(tc_sigs[:,timep], tc_rts)[0][1])

		tc_corr = np.array(tc_corr)

		pl.plot(range(int(trial_deconvolution_interval[0]*signal_sample_frequency),int(trial_deconvolution_interval[1]*signal_sample_frequency)), tc_corr)
	pl.save_figure('tc_corr.pdf')

	embed()

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_fir_signals, conditions = ['PP','UP','PU','UU'], show_legend = True, signal_labels = dict(zip(['PU','PP','UU','UP'], labels[0])), compute_mean = True, compute_sd = True)
pl.save_figure(filename='FIR_pos.pdf',sub_folder = 'over_subs')

