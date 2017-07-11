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

from joblib import Parallel, delayed
import multiprocessing

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

response_deconvolution_interval = np.array([-0.5, 4.5])
stimulus_deconvolution_interval = np.array([-0.5, 4.5])

linestylemap = {'PP': ['k-'],
				 'UP': ['b-'],
				 'PU': ['k--'],
				 'UU': ['b--']}


pl = Plotter(figure_folder = figfolder, linestylemap = linestylemap)



#### PLOT AVERAGES OVER SUBS

# pl.open_figure()

# pl.subplot(1,2,2, title= 'Average pupil difference')

power_time_window = [100,600]#[15,30]
zero_point = 15

# all_ie_scores = []
all_betas = []


FIR_signals = {'noPEc': [],
			  'bothPEc':[],
			  'TIc':[],
			  'TRc':[],
			  'noPEic':[],
			  'bothPEic':[],
			  'TIic':[],
			  'TRic':[]}

plot_indiv = False#True

#for subname in sublist:
def run_analysis(subname):





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

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

	# try:
	betas,labels = pa.get_IRF_correct_incorrect()

	for b,l in zip(betas,labels):
		FIR_signals[l].append(b)

num_cores = multiprocessing.cpu_count()


Parallel(n_jobs=8)(delayed(run_analysis)(subname) for subname in sublist)

embed()