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

#sublist = ['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN']#
# sublist = ['AA','AB','AC','AD','AF','AG','AH','AI','AJ','AM']
sublist = ['AA','AB','AC','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS']#,'DA','DB','DC','DD','DE','DF']#
# sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AD','AE','AK','AL','AM','AN']
sbsetting = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
response_deconvolution_interval = np.array([-2, 5])
stimulus_deconvolution_interval = np.array([-2, 2])

down_fs = 100

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


stimulus_fir_signals = {'color': {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []},
						'ori': {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []}
						 }

response_fir_signals = {'color': {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []},
						'ori': {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []}
						 }



for subname in sublist:



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

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

	betas, labels = pa.get_IRF()

	# embed()


	stimulus_fir_signals['color']['PU'].append(betas[0][:,0])
	stimulus_fir_signals['color']['PP'].append(betas[0][:,1])
	stimulus_fir_signals['color']['UU'].append(betas[0][:,2])
	stimulus_fir_signals['color']['UP'].append(betas[0][:,3])

	stimulus_fir_signals['ori']['PU'].append(betas[1][:,0])
	stimulus_fir_signals['ori']['PP'].append(betas[1][:,1])
	stimulus_fir_signals['ori']['UU'].append(betas[1][:,2])
	stimulus_fir_signals['ori']['UP'].append(betas[1][:,3])	

	response_fir_signals['color']['PU'].append(betas[3][:,0])
	response_fir_signals['color']['PP'].append(betas[3][:,1])
	response_fir_signals['color']['UU'].append(betas[3][:,2])
	response_fir_signals['color']['UP'].append(betas[3][:,3])

	response_fir_signals['ori']['PU'].append(betas[4][:,0])
	response_fir_signals['ori']['PP'].append(betas[4][:,1])
	response_fir_signals['ori']['UU'].append(betas[4][:,2])
	response_fir_signals['ori']['UP'].append(betas[4][:,3])	


	# recorded_signal = pa.resampled_pupil_signal#pa.read_pupil_data(pa.combined_h5_filename, signal_type = 'long_signal')
	# predicted_signal = np.dot(pa.fir_betas.T.astype(float32), pa.design_matrix.astype(float32))
	# r_squared = 1.0 - ((predicted_signal.T -recorded_signal)**2).sum(axis=-1) / (recorded_signal**2).sum(axis=-1)

	# plt.figure()
	# plt.title('R^2 = %.2f'%r_squared)
	# plt.plot(recorded_signal, label='pupil_signal')
	# plt.plot(predicted_signal, label='predicted_signal')
	# plt.legend()

	# sn.despine()
	# plt.savefig(os.path.join(figfolder, 'per_sub','FIR','%s-timecourse.pdf'%subname))

	plt.figure()
	ax=plt.subplot(2,2,1)
	plt.title('Stimulus-locked - color')
	plt.plot(betas[0]-betas[0][:5,:].mean(axis=0))
	ax.set(xticks = np.arange(0,45,5), xticklabels = np.arange(-.5,4,0.5))
	plt.legend(labels[1])

	# ax.set(xticks=np.arange(0,160,20), xticklabels=np.arange(-2,6))

	sn.despine(offset=5)

	ax=plt.subplot(2,2,2)
	plt.title('Response-locked - color')
	plt.plot(betas[3]-betas[3][:5,:].mean(axis=0))
	ax.set(xticks = np.arange(0,50,5), xticklabels = np.arange(-2,3,0.5))
	plt.legend(labels[3])

	# ax.set(xticks=np.arange(0,160,20), xticklabels=np.arange(-2,6))

	sn.despine(offset=5)

	ax=plt.subplot(2,2,3)
	plt.title('Stimulus-locked - ori')
	plt.plot(betas[1]-betas[1][:5,:].mean(axis=0))
	ax.set(xticks = np.arange(0,45,5), xticklabels = np.arange(-.5,4,0.5))
	plt.legend(labels[2])

	# ax.set(xticks=np.arange(0,160,20), xticklabels=np.arange(-2,6))

	sn.despine(offset=5)	

	ax=plt.subplot(2,2,4)
	plt.title('Response-locked - ori')
	plt.plot(betas[4]-betas[4][:5,:].mean(axis=0))
	ax.set(xticks = np.arange(0,50,5), xticklabels = np.arange(-2,3,0.5))
	plt.legend(labels[4])

	# ax.set(xticks=np.arange(0,160,20), xticklabels=np.arange(-2,6))

	sn.despine(offset=5)	

	plt.tight_layout()

	plt.savefig(os.path.join(figfolder,'per_sub','FIR','%s-FIR.pdf'%subname))

	plt.close()

# embed()



plt.figure()

all_data_ndarray = np.dstack([stimulus_fir_signals['color']['PU'],stimulus_fir_signals['color']['PP'],stimulus_fir_signals['color']['UU'],stimulus_fir_signals['color']['UP']])
ax=plt.subplot(2,2,1)
ax.title('Stimulus-locked - color')
ax.ylabel(r'Pupil size ($\beta$)')
ax.axvline(x=0, color='k', linestyle='solid', alpha=0.15)
ax.axhline(y=0, color='k', linestyle='dashed', alpha=0.25)

sn.tsplot(data = all_data_ndarray, condition = labels[0], time = pd.Series(data=np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1], 1/deconv_sample_frequency), name= 'Time(s)'), ci=[68], legend=True)

sn.despine(offset=5)

all_data_ndarray = np.dstack([response_fir_signals['color']['PU'],response_fir_signals['color']['PP'],response_fir_signals['color']['UU'],response_fir_signals['color']['UP']])
ax=plt.subplot(2,2,2)
ax.title('Response-locked - color')
ax.ylabel(r'Pupil size ($\beta$)')
ax.axvline(x=0, color='k', linestyle='solid', alpha=0.15)
ax.axhline(y=0, color='k', linestyle='dashed', alpha=0.25)

sn.tsplot(data = all_data_ndarray, condition = labels[0], time = pd.Series(data=np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1], 1/deconv_sample_frequency), name= 'Time(s)'), ci=[68], legend=True)

sn.despine(offset=5)

all_data_ndarray = np.dstack([stimulus_fir_signals['ori']['PU'],stimulus_fir_signals['ori']['PP'],stimulus_fir_signals['ori']['UU'],stimulus_fir_signals['ori']['UP']])
ax=plt.subplot(2,2,3)
ax.title('Stimulus-locked - ori')
ax.ylabel(r'Pupil size ($\beta$)')
ax.axvline(x=0, color='k', linestyle='solid', alpha=0.15)
ax.axhline(y=0, color='k', linestyle='dashed', alpha=0.25)

sn.tsplot(data = all_data_ndarray, condition = labels[0], time = pd.Series(data=np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1], 1/deconv_sample_frequency), name= 'Time(s)'), ci=[68], legend=True)

sn.despine(offset=5)

all_data_ndarray = np.dstack([response_fir_signals['ori']['PU'],response_fir_signals['ori']['PP'],response_fir_signals['ori']['UU'],response_fir_signals['ori']['UP']])
ax=plt.subplot(2,2,4)
ax.title('Response-locked - ori')
ax.ylabel(r'Pupil size ($\beta$)')
ax.axvline(x=0, color='k', linestyle='solid', alpha=0.15)
ax.axhline(y=0, color='k', linestyle='dashed', alpha=0.25)

sn.tsplot(data = all_data_ndarray, condition = labels[0], time = pd.Series(data=np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1], 1/deconv_sample_frequency), name= 'Time(s)'), ci=[68], legend=True)

sn.despine(offset=5)

plt.savefig(os.path.join(figfolder,'over_subs','FIR_all.pdf'))

plt.close()
# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = response_fir_signals, conditions = ['PP','UP','PU','UU'], show_legend = True, signal_labels = dict(zip(['PU','PP','UU','UP'], labels[0])), compute_mean = True, compute_sd = True)
# pl.save_figure(filename='FIR_all.pdf',sub_folder = 'over_subs')


# for subname in sublist_neg:



# 	stimulus_pupil_signals = {'PP': [],
# 					 'UP': [],
# 					 'PU': [],
# 					 'UU': []}				 
# 	response_diff_signals  = {'PP': [],
# 					 'UP': [],
# 					 'PU': [],
# 					 'UU': []}	
# 	stimulus_diff_signals  = {'PP': [],
# 					 'UP': [],
# 					 'PU': [],
# 					 'UU': []}				 
# 	power_signals = {'PP': [],
# 					 'UP': [],
# 					 'PU': [],
# 					 'UU': []}
# 	ie_scores 	  = {'PP': [],
# 					 'UP': [],
# 					 'PU': [],
# 					 'UU': []}
# 	rts 	  	  = {'PP': [],
# 					 'UP': [],
# 					 'PU': [],
# 					 'UU': []}
# 	pc 	  	  	  = {'PP': [],
# 					 'UP': [],
# 					 'PU': [],
# 					 'UU': []}			
# 	all_ie_scores = {'PP': [],
# 					 'UP': [],
# 					 'PU': [],
# 					 'UU': []}	

# 	this_sub_IRF = {'stimulus': [], 'button_press': []}

# 	# print subname
# 	# Organize filenames
# 	rawfolder = os.path.join(raw_data_folder,subname)
# 	sharedfolder = os.path.join(shared_data_folder,subname)
# 	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
# 	h5filename = os.path.join(sharedfolder,subname+'.h5')

# 	#pl.figure_folder = os.path.join(rawfolder,'results/')

# 	# if not os.path.isdir(os.path.join(rawfolder,'results/')):
# 	# 	os.makedirs(os.path.join(rawfolder,'results/'))

# 	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

# 	betas, labels = pa.get_IRF()

# 	pe_betas = betas[0]
# 	other_betas = betas[1]

# 	response_fir_signals['PU'].append(pe_betas[:,0])
# 	response_fir_signals['PP'].append(pe_betas[:,1])
# 	response_fir_signals['UU'].append(pe_betas[:,2])
# 	response_fir_signals['UP'].append(pe_betas[:,3])

# 	recorded_signal = pa.resampled_pupil_signal#pa.read_pupil_data(pa.combined_h5_filename, signal_type = 'long_signal')
# 	predicted_signal = np.dot(pa.fir_betas.T.astype(float32), pa.design_matrix.astype(float32))

# 	plt.figure()
# 	plt.plot(recorded_signal, label='pupil_signal')
# 	plt.plot(predicted_signal, label='predicted_signal')
# 	plt.legend()

# 	sn.despine()
# 	plt.savefig(os.path.join(figfolder, 'per_sub','FIR','%s-timecourse.pdf'%subname))

# 	plt.figure()
# 	ax=plt.subplot(1,2,1)
# 	plt.title('Nuissances')
# 	plt.plot(other_betas)
# 	plt.legend(labels[1])

# 	# ax.set(xticks=np.arange(0,160,20), xticklabels=np.arange(-2,6))

# 	sn.despine()

# 	ax=plt.subplot(1,2,2)
# 	plt.title('PE')
# 	plt.plot(pe_betas-pe_betas[:5,:].mean(axis=0))
# 	plt.legend(labels[0])
# 	# ax.set(xticks=np.arange(0,100,20), xticklabels=np.arange(-1,4))

# 	sn.despine()

# 	plt.tight_layout()

# 	plt.savefig(os.path.join(figfolder,'per_sub','FIR','%s-FIR.pdf'%subname))

# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = response_fir_signals, conditions = ['PP','UP','PU','UU'], show_legend = True, signal_labels = dict(zip(['PU','PP','UU','UP'], labels[0])), compute_mean = True, compute_sd = True)
# pl.save_figure(filename='FIR_neg.pdf',sub_folder = 'over_subs')