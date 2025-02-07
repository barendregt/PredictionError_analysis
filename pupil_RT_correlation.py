

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


from analysis_parameters import *
# linestylemap = {'PP': ['k-'],
# 				 'UP': [''],
# 				 'PU': [''],
# 				 'UU': ['']}

pl = Plotter(figure_folder = figfolder)



all_correlations = {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []}

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

	#pa.recombine_signal_blocks(force_rebuild=True)

	recorded_pupil_signal = pa.read_pupil_data(pa.combined_h5_filename, signal_type='clean_signal')

	resampled_pupil_signal = sp.signal.resample(recorded_pupil_signal, int((recorded_pupil_signal.shape[-1] / signal_sample_frequency)*deconv_sample_frequency), axis = -1)

	trial_params = pa.read_trial_data(pa.combined_h5_filename)

	tcodes = [0,10,30,50,70]
	tnames = ['noPE','PEtr','PEntr','bothPE']

	# pl.open_figure(force=1)
	# pl.hline(y=0)
	# pl.vline(x=np.abs(trial_deconvolution_interval[0])*deconv_sample_frequency)

	tc_correlations = dict(list(zip(tnames,[[]]*4)))
	for tcii in range(len(tcodes)-1):
		
		tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==1)*(trial_params['trial_codes'] >= tcodes[tcii]) * (trial_params['trial_codes'] < tcodes[tcii+1])]

		trials = tc_rts + trial_params['trial_phase_7_full_signal'][(trial_params['correct_answer']==1)*(trial_params['trial_codes'] >= tcodes[tcii]) * (trial_params['trial_codes'] < tcodes[tcii+1])] / signal_sample_frequency*deconv_sample_frequency

		tc_sigs = []
		for trii,trial in enumerate(trials):
			if (trial+(trial_deconvolution_interval[1]*deconv_sample_frequency)) <= resampled_pupil_signal.shape[0]:
				tc_sigs.append(resampled_pupil_signal[int(trial+(trial_deconvolution_interval[0]*deconv_sample_frequency)):int(trial+(trial_deconvolution_interval[1]*deconv_sample_frequency))])
			# else:
			# 	tc_rts = tc_rts.delete(trii)
		tc_sigs = np.vstack(tc_sigs)
		tc_rts = tc_rts[:tc_sigs.shape[0]]
		tc_corr = []
		for timep in range(tc_sigs.shape[1]):
			tc_corr.append(np.corrcoef(tc_sigs[:,timep], tc_rts)[0][1])

		tc_correlations[tnames[tcii]] = np.array(tc_corr)
		all_correlations[['PP','UP','PU','UU'][tcii]].append(np.array(tc_corr))

	
	# pl.event_related_pupil_average(data = tc_correlations, conditions = tnames, show_legend = True)
	# pl.save_figure('%s-tc_corr_response.pdf'%subname, sub_folder='per_sub/RT')
	# plt.close('all')
	# embed()

# embed()

all_data_ndarray = np.dstack([all_correlations['PU'],all_correlations['PP'],all_correlations['UU'],all_correlations['UP']])

plt.figure()

plt.ylabel(r'Pupil-RT correlation ($r$)')
plt.axvline(x=0, color='k', linestyle='solid', alpha=0.15)
plt.axhline(y=0, color='k', linestyle='dashed', alpha=0.25)

sn.tsplot(data = all_data_ndarray, condition = tnames, time = pd.Series(data=np.arange(trial_deconvolution_interval[0], trial_deconvolution_interval[1], 1/deconv_sample_frequency), name= 'Time(s)'), ci=[68], legend=True)

sn.despine(offset=5)
plt.savefig(os.path.join(figfolder,'over_subs','RT_all_correct.pdf'))
plt.close()
# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.vline(x=np.abs(trial_deconvolution_interval[0])*deconv_sample_frequency)
# pl.event_related_pupil_average(data = all_correlations, conditions = ['PP','UP','PU','UU'], show_legend = True, ylabel = 'Correlation (r)', signal_labels = dict(zip(['PU','PP','UU','UP'], tnames)), xticks = np.arange(0, all_correlations['PP'][0].shape[0], int(all_correlations['PP'][0].shape[0]/deconv_sample_frequency)), xticklabels = np.arange(trial_deconvolution_interval[0], trial_deconvolution_interval[1]+1, .5), compute_mean = True, compute_sd = True)
# pl.save_figure(filename='tc_corr_response.pdf',sub_folder = 'over_subs/RT')

all_correlations = {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []}

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

	#pa.recombine_signal_blocks(force_rebuild=True)

	recorded_pupil_signal = pa.read_pupil_data(pa.combined_h5_filename, signal_type='clean_signal')

	resampled_pupil_signal = sp.signal.resample(recorded_pupil_signal, int((recorded_pupil_signal.shape[-1] / signal_sample_frequency)*deconv_sample_frequency), axis = -1)

	trial_params = pa.read_trial_data(pa.combined_h5_filename)

	tcodes = [0,10,30,50,70]
	tnames = ['noPE','PEtr','PEntr','bothPE']

	# pl.open_figure(force=1)
	# pl.hline(y=0)
	# pl.vline(x=np.abs(trial_deconvolution_interval[0])*deconv_sample_frequency)

	tc_correlations = dict(list(zip(tnames,[[]]*4)))
	for tcii in range(len(tcodes)-1):
		
		tc_rts = trial_params['reaction_time'][(trial_params['correct_answer']==0)*(trial_params['trial_codes'] >= tcodes[tcii]) * (trial_params['trial_codes'] < tcodes[tcii+1])]

		trials = tc_rts + trial_params['trial_phase_7_full_signal'][(trial_params['correct_answer']==0)*(trial_params['trial_codes'] >= tcodes[tcii]) * (trial_params['trial_codes'] < tcodes[tcii+1])] / signal_sample_frequency*deconv_sample_frequency

		tc_sigs = []
		for trii,trial in enumerate(trials):
			if (trial+(trial_deconvolution_interval[1]*deconv_sample_frequency)) <= resampled_pupil_signal.shape[0]:
				tc_sigs.append(resampled_pupil_signal[int(trial+(trial_deconvolution_interval[0]*deconv_sample_frequency)):int(trial+(trial_deconvolution_interval[1]*deconv_sample_frequency))])
			# else:
			# 	tc_rts = tc_rts.delete(trii)
		tc_sigs = np.vstack(tc_sigs)
		tc_rts = tc_rts[:tc_sigs.shape[0]]
		tc_corr = []
		for timep in range(tc_sigs.shape[1]):
			tc_corr.append(np.corrcoef(tc_sigs[:,timep], tc_rts)[0][1])

		tc_correlations[tnames[tcii]] = np.array(tc_corr)
		all_correlations[['PP','UP','PU','UU'][tcii]].append(np.array(tc_corr))

	
	# pl.event_related_pupil_average(data = tc_correlations, conditions = tnames, show_legend = True)
	# pl.save_figure('%s-tc_corr_response.pdf'%subname, sub_folder='per_sub/RT')
	# plt.close('all')
	# embed()

# embed()

all_data_ndarray = np.dstack([all_correlations['PU'],all_correlations['PP'],all_correlations['UU'],all_correlations['UP']])

plt.figure()

plt.ylabel(r'Pupil-RT correlation ($r$)')
plt.axvline(x=0, color='k', linestyle='solid', alpha=0.15)
plt.axhline(y=0, color='k', linestyle='dashed', alpha=0.25)

sn.tsplot(data = all_data_ndarray, condition = tnames, time = pd.Series(data=np.arange(trial_deconvolution_interval[0], trial_deconvolution_interval[1], 1/deconv_sample_frequency), name= 'Time(s)'), ci=[68], legend=True)

sn.despine(offset=5)
plt.savefig(os.path.join(figfolder,'over_subs','RT_all_incorrect.pdf'))
plt.close()