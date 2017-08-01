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

for subname in sublist:
# def run_analysis(subname):





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
	subFIR_signals = pickle.load(open(os.path.join(raw_data_folder,'FIR_data','%s_FIR_correct_incorrect.pickle'%subname),'rb'))

	for f in FIR_signals.subkeys():
	sub	FIR_signals[f] = FIR_signals[f][0]



	plt.figure()
	a=plt.subplot(1,2,1)
	plt.title('Correct')

	subpd.DataFrame.from_dict(FIR_signals)[['TIc','TRc','bothPEc','noPEc']].plot(ax=a)

	a=plt.subplot(1,2,2)
	plt.title('Incorrect')

	subpd.DataFrame.from_dict(FIR_signals)[['TIic','TRic','bothPEic','noPEic']].plot(ax=a)	

	plt.tight_layout()

	sn.despine(offset=5)

	plt.savefig(os.path.join(figfolder,'per_sub','FIR','%s_fir.pdf'%subname))



	plt.close()
	# embed()
