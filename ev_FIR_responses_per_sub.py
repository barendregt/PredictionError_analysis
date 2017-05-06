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
sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO']
# sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AD','AE','AK','AL','AM','AN']
sbsetting = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

low_pass_pupil_f, high_pass_pupil_f = 4.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
response_deconvolution_interval = np.array([-2, 7])
stimulus_deconvolution_interval = np.array([-1, 2])

down_fs = 100

pl = Plotter(figure_folder = figfolder)



#### PLOT AVERAGES OVER SUBS

# pl.open_figure()

# pl.subplot(1,2,2, title= 'Average pupil difference')

power_time_window = [100,600]#[15,30]
zero_point = 15

# all_ie_scores = []
all_rts = []

for subname in sublist:


	response_pupil_signals = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}
	stimulus_pupil_signals = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}				 
	response_diff_signals  = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}	
	stimulus_diff_signals  = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}				 
	power_signals = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}
	ie_scores 	  = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}
	rts 	  	  = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}
	pc 	  	  	  = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}			
	all_ie_scores = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}	

	this_sub_IRF = {'stimulus': [], 'button_press': []}

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	#pl.figure_folder = os.path.join(rawfolder,'results/')

	# if not os.path.isdir(os.path.join(rawfolder,'results/')):
	# 	os.makedirs(os.path.join(rawfolder,'results/'))

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, sort_by_date = sbsetting[sublist.index(subname)], signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

	betas, labels = pa.get_IRF()

	pe_betas = dict(zip(labels[0], betas[0]))
	other_betas = dict(zip(labels[1], betas[1]))

	pl.open_figure(force=1)
	pl.subplot(1,2,1)
	pl.hline(y=0)
	pl.event_related_pupil_average(data = pe_betas, conditions=labels[0], signal_labels = {'noPE': 'Predicted', 'PEtr': 'Task relevant','PEntr':'Task irrelevant','bothPE':'Both'}, show_legend=True, ylabel = 'Pupil size')#, x_lim = [0, 3*(signal_sample_frequency/down_fs)], xticks = np.arange(0,3*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5))

	#pl.open_figure(force=1)
	pl.subplot(1,2,2)
	pl.hline(y=0)
	pl.event_related_pupil_average(data = other_betas, conditions=labels[1], show_legend=True, ylabel = 'Pupil size')#, x_lim = [0, 9*(signal_sample_frequency/down_fs)], xticks = np.arange(0,9*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5))

	pl.save_figure('%s-FIR.pdf'%subname, sub_folder='per_sub/FIR')