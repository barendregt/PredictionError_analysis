from __future__ import division

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
# import readline
# from rpy2.robjects.packages import importr
# import rpy2.robjects as ro
# import pandas.rpy.common as com
# stats = importr('stats')
# base = importr('base')

from math import *
import os,glob,sys,platform

import cPickle as pickle
import pandas as pd

from IPython import embed

# sys.path.append('tools/')
from BehaviorAnalyzer import BehaviorAnalyzer
from Plotter import Plotter


from analysis_parameters import *

pl = Plotter(figure_folder = figfolder, linestylemap = linestylemap)



#### PLOT AVERAGES OVER SUBS

# pl.open_figure()

# pl.subplot(1,2,2, title= 'Average pupil difference')

correct_response_pupil_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}
correct_stimulus_pupil_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}				 

incorrect_response_pupil_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}
incorrect_stimulus_pupil_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}	

correct_response_diff_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}
correct_stimulus_diff_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}	

incorrect_response_diff_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}
incorrect_stimulus_diff_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}	


correct_response_diff_subids = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}
correct_stimulus_diff_subids = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}	

incorrect_response_diff_subids = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}
incorrect_stimulus_diff_subids = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}	


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
all_rts 	  = {'PP': [],
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
dp 	  	  	  = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}	
bias  	  	  = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}					 				 		
all_ie_scores = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}	

all_sub_IRF = {'stimulus': [], 'button_press': []}

power_time_window = [int(0.5*(signal_sample_frequency/down_fs)),int(3*(signal_sample_frequency/down_fs))]#[15,30]
zero_point = 15

# all_ie_scores = []
# all_rts = []

for subname in sublist:

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

	# redo signal extraction
	# pa.recombine_signal_blocks(force_rebuild = True)

	#sub_rts = pa.compute_reaction_times()

	# # Get pupil data (ev)
	pa.signal_per_trial(only_correct = True, only_incorrect = False, reference_phase = 7, with_rt = True, baseline_type = 'relative', baseline_period = [-0.5, 0.0], force_rebuild=False, down_sample = True)

	# pa.get_IRF()

	# all_sub_IRF['button_press'].append(pa.sub_IRF['button_press'])


	# embed()
	ref_signals = []

	for key,trial_signal in pa.trial_signals.items():
		if key < 10:
			#trial_signal = trial_signal - trial_signal[:,zero_point][:,np.newaxis]

			ref_signals.extend(trial_signal)

	msignal = np.mean(ref_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	pp_signal = []
	up_signal = []
	pu_signal = []
	uu_signal = []
	

	try:
		for key, trial_signal in pa.trial_signals.items():
			if key < 10:
				response_pupil_signals['PP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
				correct_response_pupil_signals['PP'].append(np.mean(trial_signal, axis=0))
			elif key < 30:
				response_pupil_signals['PU'].extend(trial_signal)#.extend(trial_signal)
				correct_response_pupil_signals['PU'].append(np.mean(trial_signal, axis=0))
			elif key < 50:
				response_pupil_signals['UP'].extend(trial_signal)#.extend(trial_signal)
				correct_response_pupil_signals['UP'].append(np.mean(trial_signal, axis=0))
			else:
				response_pupil_signals['UU'].extend(trial_signal)#.extend(trial_signal)
				correct_response_pupil_signals['UU'].append(np.mean(trial_signal, axis=0))

			if len(trial_signal)>0:

				if key < 10:
					response_diff_signals['PP'].extend(trial_signal - msignal)
					correct_response_diff_signals['PP'].extend(trial_signal - msignal)
					correct_response_diff_subids['PP'].extend([subname]*trial_signal.shape[0])
				elif key < 30:
					response_diff_signals['PU'].extend(trial_signal - msignal)
					correct_response_diff_signals['PU'].extend(trial_signal - msignal)
					correct_response_diff_subids['PU'].extend([subname]*trial_signal.shape[0])
				elif key < 50:
					response_diff_signals['UP'].extend(trial_signal - msignal)
					correct_response_diff_signals['UP'].extend(trial_signal - msignal)
					correct_response_diff_subids['UP'].extend([subname]*trial_signal.shape[0])
				else:
					response_diff_signals['UU'].extend(trial_signal - msignal)
					correct_response_diff_signals['UU'].extend(trial_signal - msignal)
					correct_response_diff_subids['UU'].extend([subname]*trial_signal.shape[0])
	except:
		embed()

	# ref_signals = []

	# for key,trial_signal in pa.trial_signals.items():

	# 	if key < 10:
	# 		trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]]# - trial_signal[:,zero_point][:,np.newaxis]

	# 		ref_signals.extend(trial_signal)

	# msignal = np.mean(ref_signals, axis=0)
	# msignal_norm = np.linalg.norm(msignal, ord=2)**2

	# for key,trial_signal in pa.trial_signals.items():
	# 	trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]]# - trial_signal[:,zero_point][:,np.newaxis]

	# 	power_signal = np.dot(trial_signal, msignal)/msignal_norm

	# 	if key < 10:
	# 		pass
	# 		# power_signals['PP'].extend(power_signal)
	# 	if key < 30:
	# 		power_signals['UP'].extend(power_signal)
	# 	elif key < 50:
	# 		power_signals['UU'].extend(power_signal)
	# 	else:
	# 		power_signals['PU'].extend(power_signal)

	pa.deconvolution_interval = stimulus_deconvolution_interval
	pa.signal_per_trial(only_correct = True, only_incorrect = False, reference_phase = 4, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = True)

	# pa.get_IRF()

	# all_sub_IRF['stimulus'].append(pa.sub_IRF['stimulus'])

	# embed()
	ref_signals = []

	for key,trial_signal in pa.trial_signals.items():
		if key < 10:
			#trial_signal = trial_signal - trial_signals[:,zero_point][:,np.newaxis]

			ref_signals.extend(trial_signal)

	msignal = np.mean(ref_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	pp_signal = []
	up_signal = []
	pu_signal = []
	uu_signal = []
	

	for key, trial_signal in pa.trial_signals.items():
		if key < 10:
			stimulus_pupil_signals['PP'].extend(trial_signal)
			correct_stimulus_pupil_signals['PP'].append(np.mean(trial_signal, axis=0))
		elif key < 30:
			stimulus_pupil_signals['PU'].extend(trial_signal)
			correct_stimulus_pupil_signals['PU'].append(np.mean(trial_signal, axis=0))
		elif key < 50:
			stimulus_pupil_signals['UP'].extend(trial_signal)
			correct_stimulus_pupil_signals['UP'].append(np.mean(trial_signal, axis=0))
		else:
			stimulus_pupil_signals['UU'].extend(trial_signal)
			correct_stimulus_pupil_signals['UU'].append(np.mean(trial_signal, axis=0))

		if len(trial_signal)>0:
			if key < 10:
				stimulus_diff_signals['PP'].extend(trial_signal - msignal)
				correct_stimulus_diff_signals['PP'].extend(trial_signal - msignal)
				correct_stimulus_diff_subids['PP'].extend([subname]*trial_signal.shape[0])
			elif key < 30:
				stimulus_diff_signals['PU'].extend(trial_signal - msignal)
				correct_stimulus_diff_signals['PU'].extend(trial_signal - msignal)
				correct_stimulus_diff_subids['PU'].extend([subname]*trial_signal.shape[0])
			elif key < 50:
				stimulus_diff_signals['UP'].extend(trial_signal - msignal)
				correct_stimulus_diff_signals['UP'].extend(trial_signal - msignal)
				correct_stimulus_diff_subids['UP'].extend([subname]*trial_signal.shape[0])
			else:
				stimulus_diff_signals['UU'].extend(trial_signal - msignal)
				correct_stimulus_diff_signals['UU'].extend(trial_signal - msignal)
				correct_stimulus_diff_subids['UU'].extend([subname]*trial_signal.shape[0])

	# sub_rts = pa.compute_reaction_times()

	# all_rts['PP'].append(np.median(sub_rts[0]))
	# all_rts['PP'].append(np.median(sub_rts[1]))
	# all_rts['UP'].append(np.median(sub_rts[10]))
	# all_rts['UP'].append(np.median(sub_rts[20]))
	# all_rts['PU'].append(np.median(sub_rts[30]))
	# all_rts['PU'].append(np.median(sub_rts[40]))
	# all_rts['UU'].append(np.median(sub_rts[50]))
	# all_rts['UU'].append(np.median(sub_rts[60]))

	# rts['UP'].append(np.mean(sub_rts[10] / np.median(sub_rts[0])))
	# rts['UP'].append(np.mean(sub_rts[20] / np.median(sub_rts[1])))
	# rts['PU'].append(np.mean(sub_rts[30] / np.median(sub_rts[0])))
	# rts['PU'].append(np.mean(sub_rts[40] / np.median(sub_rts[1])))
	# rts['UU'].append(np.mean(sub_rts[50] / np.median(sub_rts[0])))
	# rts['UU'].append(np.mean(sub_rts[60] / np.median(sub_rts[1])))


	# sub_pc = pa.compute_percent_correct()

	# pc['PP'].append(sub_pc[0])
	# pc['PP'].append(sub_pc[1])
	# pc['UP'].append(sub_pc[10])
	# pc['UP'].append(sub_pc[20])
	# pc['PU'].append(sub_pc[30])
	# pc['PU'].append(sub_pc[40])
	# pc['UU'].append(sub_pc[50])
	# pc['UU'].append(sub_pc[60])

	# sub_dp, sub_bias = pa.compute_dprime()
	# dp['PP'].append(sub_dp[0])
	# dp['PP'].append(sub_dp[1])
	# dp['UP'].append(sub_dp[10])
	# dp['UP'].append(sub_dp[20])
	# dp['PU'].append(sub_dp[30])
	# dp['PU'].append(sub_dp[40])
	# dp['UU'].append(sub_dp[50])
	# dp['UU'].append(sub_dp[60])	

	# bias['PP'].append(sub_bias[0])
	# bias['PP'].append(sub_bias[1])
	# bias['UP'].append(sub_bias[10])
	# bias['UP'].append(sub_bias[20])
	# bias['PU'].append(sub_bias[30])
	# bias['PU'].append(sub_bias[40])
	# bias['UU'].append(sub_bias[50])
	# bias['UU'].append(sub_bias[60])

# # embed()

# pl.open_figure(force=1)
# # pl.figure.suptitle('Pupil amplitude')
# # pl.subplot(1,2,1, title='Pupil amplitude')
# pl.bar_plot(data = power_signals, conditions = ['UP','PU','UU'], with_error = True, ylabel = 'Pupil amplitude (a.u.)', x_lim = [0.5, None], y_lim = [0.6, 1.21], yticks = np.arange(0.6, 2.0, 0.2), yticklabels = np.arange(0.8,2.2,0.2), xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error')

# pl.save_figure('pupil_amplitude_bar.pdf', sub_folder = 'over_subs')

# # # Convert IE scores back to milliseconds
# # for (key,item) in ie_scores.items():
# # 	ie_scores[key] = np.array(ie_scores[key]) * np.median(all_ie_scores['PP'])
# # 	# rts[key] = np.array(rts[key]) * np.median(all_rts)

# # # pl.subplot(1,2,2, title='Reaction times')
# # # pl.open_figure(force=1)
# # # # pl.figure.suptitle('Inverse Efficiency')
# # # pl.hline(y = np.median(all_ie_scores['PP']), label = 'Predicted')
# # # pl.bar_plot(data = ie_scores, conditions = ['UP','PU','UU'], ylabel='Corrected reaction time (ms)', with_error = True, x_lim = [0.5, None], xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error category', y_lim = [1.4, None], yticks = np.arange(0.0,1.8,.1), yticklabels = np.arange(0,1800,100))

# pl.open_figure(force=1)
# # pl.hline(y=0)
# # pl.event_related_pupil_average(data = response_diff_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','UU':'Task irrelevant','PU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0]+0.5, response_deconvolution_interval[1],.5), compute_mean = True, compute_sd = True)

# # pl.save_figure('pupil_amplitude_button-press.pdf', sub_folder = 'over_subs')

# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = response_pupil_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'anova')

# pl.save_figure('pupil_response_button-press.pdf', sub_folder = 'over_subs/pupil')

# # pl.open_figure(force=1)
# # pl.hline(y=0)
# # pl.event_related_pupil_difference(data = response_pupil_signals, conditions = ['PP','UP','PU','UU'], show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs),  4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,4.5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5))

# # embed()


# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = response_diff_signals, conditions = ['UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'ttest')

# pl.save_figure('pupil_difference_button-press.pdf', sub_folder = 'over_subs/pupil')
# # pl.save_figure('pupil_amplitude-stimulus.pdf', sub_folder = 'over_subs/pupil')

# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = stimulus_pupil_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'anova')

# pl.save_figure('pupil_response-stimulus.pdf', sub_folder = 'over_subs/pupil')

# # pl.open_figure(force=1)
# # pl.hline(y=0)
# # pl.event_related_pupil_difference(data = stimulus_pupil_signals, conditions = ['PP','UP','PU','UU'], show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5))


# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = stimulus_diff_signals, conditions = ['UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'ttest')

# pl.save_figure('pupil_difference-stimulus.pdf', sub_folder = 'over_subs/pupil')



# # correct_response_pupil_signals = response_pupil_signals
# correct_stimulus_pupil_signals = stimulus_pupil_signals

#### PLOT AVERAGES OVER SUBS

# pl.open_figure()

# pl.subplot(1,2,2, title= 'Average pupil difference')

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
all_rts 	  = {'PP': [],
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

all_sub_IRF = {'stimulus': [], 'button_press': []}

power_time_window = [int(0.5*(signal_sample_frequency/down_fs)),int(3*(signal_sample_frequency/down_fs))]#[15,30]
zero_point = 15

# all_ie_scores = []
# all_rts = []

for subname in sublist:

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

	# redo signal extraction
	# pa.recombine_signal_blocks(force_rebuild = True)

	#sub_rts = pa.compute_reaction_times()

	# # Get pupil data (ev)
	pa.signal_per_trial(only_correct = False, only_incorrect = True, reference_phase = 7, with_rt = True, baseline_type = 'relative', baseline_period = [-0.5, 0.0], force_rebuild=False, down_sample = True)

	# pa.get_IRF()

	# all_sub_IRF['button_press'].append(pa.sub_IRF['button_press'])


	# embed()
	ref_signals = []

	for key,trial_signal in pa.trial_signals.items():
		if key < 10:
			#trial_signal = trial_signal - trial_signal[:,zero_point][:,np.newaxis]

			ref_signals.extend(trial_signal)

	msignal = np.mean(ref_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	pp_signal = []
	up_signal = []
	pu_signal = []
	uu_signal = []
	
    
	try:
		for key, trial_signal in pa.trial_signals.items():
			if len(trial_signal)>0:
				if key < 10:
					response_pupil_signals['PP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
					incorrect_response_pupil_signals['PP'].append(np.mean(trial_signal, axis=0))
				elif key < 30:
					response_pupil_signals['PU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
					incorrect_response_pupil_signals['PU'].append(np.mean(trial_signal, axis=0))
				elif key < 50:
					response_pupil_signals['UP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
					incorrect_response_pupil_signals['UP'].append(np.mean(trial_signal, axis=0))
				else:
					response_pupil_signals['UU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
					incorrect_response_pupil_signals['UU'].append(np.mean(trial_signal, axis=0))

			if len(trial_signal)>0:

				if key < 10:
					response_diff_signals['PP'].extend(trial_signal - msignal)
					incorrect_response_diff_signals['PP'].extend(trial_signal - msignal)
					incorrect_response_diff_subids['PP'].extend([subname]*trial_signal.shape[0])
				elif key < 30:
					response_diff_signals['PU'].extend(trial_signal - msignal)
					incorrect_response_diff_signals['PU'].extend(trial_signal - msignal)
					incorrect_response_diff_subids['PU'].extend([subname]*trial_signal.shape[0])
				elif key < 50:
					response_diff_signals['UP'].extend(trial_signal - msignal)
					incorrect_response_diff_signals['UP'].extend(trial_signal - msignal)
					incorrect_response_diff_subids['UP'].extend([subname]*trial_signal.shape[0])
				else:
					response_diff_signals['UU'].extend(trial_signal - msignal)
					incorrect_response_diff_signals['UU'].extend(trial_signal - msignal)
					incorrect_response_diff_subids['UU'].extend([subname]*trial_signal.shape[0])
	except:
		embed()

	# ref_signals = []

	# for key,trial_signal in pa.trial_signals.items():

	# 	if key < 10:
	# 		trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]]# - trial_signal[:,zero_point][:,np.newaxis]

	# 		ref_signals.extend(trial_signal)

	# msignal = np.mean(ref_signals, axis=0)
	# msignal_norm = np.linalg.norm(msignal, ord=2)**2

	# for key,trial_signal in pa.trial_signals.items():
	# 	trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]]# - trial_signal[:,zero_point][:,np.newaxis]

	# 	power_signal = np.dot(trial_signal, msignal)/msignal_norm

	# 	if key < 10:
	# 		pass
	# 		# power_signals['PP'].extend(power_signal)
	# 	if key < 30:
	# 		power_signals['UP'].extend(power_signal)
	# 	elif key < 50:
	# 		power_signals['UU'].extend(power_signal)
	# 	else:
	# 		power_signals['PU'].extend(power_signal)

	pa.deconvolution_interval = stimulus_deconvolution_interval
	pa.signal_per_trial(only_correct = False, only_incorrect = True, reference_phase = 4, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = True)

	# pa.get_IRF()

	# all_sub_IRF['stimulus'].append(pa.sub_IRF['stimulus'])

	# embed()
	ref_signals = []

	for key,trial_signal in pa.trial_signals.items():
		if key < 10:
			#trial_signal = trial_signal - trial_signals[:,zero_point][:,np.newaxis]

			ref_signals.extend(trial_signal)

	msignal = np.mean(ref_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	pp_signal = []
	up_signal = []
	pu_signal = []
	uu_signal = []
	

	for key, trial_signal in pa.trial_signals.items():
		if len(trial_signal)>0:
			if key < 10:
				stimulus_pupil_signals['PP'].extend(trial_signal)
				incorrect_stimulus_pupil_signals['PP'].append(np.mean(trial_signal, axis=0))
			elif key < 30:
				stimulus_pupil_signals['PU'].extend(trial_signal)
				incorrect_stimulus_pupil_signals['PU'].append(np.mean(trial_signal, axis=0))
			elif key < 50:
				stimulus_pupil_signals['UP'].extend(trial_signal)
				incorrect_stimulus_pupil_signals['UP'].append(np.mean(trial_signal, axis=0))
			else:
				stimulus_pupil_signals['UU'].extend(trial_signal)
				incorrect_stimulus_pupil_signals['UU'].append(np.mean(trial_signal, axis=0))

		if len(trial_signal)>0:
			if key < 10:
				stimulus_diff_signals['PP'].extend(trial_signal - msignal)
				incorrect_stimulus_diff_signals['PP'].extend(trial_signal - msignal)
				incorrect_stimulus_diff_subids['PP'].extend([subname]*trial_signal.shape[0])
			elif key < 30:
				stimulus_diff_signals['PU'].extend(trial_signal - msignal)
				incorrect_stimulus_diff_signals['PU'].extend(trial_signal - msignal)
				incorrect_stimulus_diff_subids['PU'].extend([subname]*trial_signal.shape[0])
			elif key < 50:
				stimulus_diff_signals['UP'].extend(trial_signal - msignal)
				incorrect_stimulus_diff_signals['UP'].extend(trial_signal - msignal)
				incorrect_stimulus_diff_subids['UP'].extend([subname]*trial_signal.shape[0])
			else:
				stimulus_diff_signals['UU'].extend(trial_signal - msignal)
				incorrect_stimulus_diff_signals['UU'].extend(trial_signal - msignal)
				incorrect_stimulus_diff_subids['UU'].extend([subname]*trial_signal.shape[0])

	# sub_rts = pa.compute_reaction_times(correct_trials = False)

	# all_rts['PP'].append(np.median(sub_rts[0]))
	# all_rts['PP'].append(np.median(sub_rts[1]))
	# all_rts['UP'].append(np.median(sub_rts[10]))
	# all_rts['UP'].append(np.median(sub_rts[20]))
	# all_rts['PU'].append(np.median(sub_rts[30]))
	# all_rts['PU'].append(np.median(sub_rts[40]))
	# all_rts['UU'].append(np.median(sub_rts[50]))
	# all_rts['UU'].append(np.median(sub_rts[60]))

	# rts['UP'].append(np.mean(sub_rts[10] / np.median(sub_rts[0])))
	# rts['UP'].append(np.mean(sub_rts[20] / np.median(sub_rts[1])))
	# rts['PU'].append(np.mean(sub_rts[30] / np.median(sub_rts[0])))
	# rts['PU'].append(np.mean(sub_rts[40] / np.median(sub_rts[1])))
	# rts['UU'].append(np.mean(sub_rts[50] / np.median(sub_rts[0])))
	# rts['UU'].append(np.mean(sub_rts[60] / np.median(sub_rts[1])))


	# sub_pc = pa.compute_percent_correct()

	# pc['PP'].append(sub_pc[0])
	# pc['PP'].append(sub_pc[1])
	# pc['UP'].append(sub_pc[10])
	# pc['UP'].append(sub_pc[20])
	# pc['PU'].append(sub_pc[30])
	# pc['PU'].append(sub_pc[40])
	# pc['UU'].append(sub_pc[50])
	# pc['UU'].append(sub_pc[60])

# # embed()

# pl.open_figure(force=1)
# # pl.figure.suptitle('Pupil amplitude')
# # pl.subplot(1,2,1, title='Pupil amplitude')
# pl.bar_plot(data = power_signals, conditions = ['UP','PU','UU'], with_error = True, ylabel = 'Pupil amplitude (a.u.)', x_lim = [0.5, None], y_lim = [0.6, 1.21], yticks = np.arange(0.6, 2.0, 0.2), yticklabels = np.arange(0.8,2.2,0.2), xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error')

# pl.save_figure('pupil_amplitude_bar.pdf', sub_folder = 'over_subs')

# # # Convert IE scores back to milliseconds
# # for (key,item) in ie_scores.items():
# # 	ie_scores[key] = np.array(ie_scores[key]) * np.median(all_ie_scores['PP'])
# # 	# rts[key] = np.array(rts[key]) * np.median(all_rts)

# # # pl.subplot(1,2,2, title='Reaction times')
# # # pl.open_figure(force=1)
# # # # pl.figure.suptitle('Inverse Efficiency')
# # # pl.hline(y = np.median(all_ie_scores['PP']), label = 'Predicted')
# # # # pl.bar_plot(data = ie_scores, conditions = ['UP','PU','UU'], ylabel='Corrected reaction time (ms)', with_error = True, x_lim = [0.5, None], xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error category', y_lim = [1.4, None], yticks = np.arange(0.0,1.8,.1), yticklabels = np.arange(0,1800,100))

# # pl.open_figure(force=1)
# # pl.hline(y=0)
# # pl.event_related_pupil_average(data = response_diff_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','UU':'Task irrelevant','PU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0]+0.5, response_deconvolution_interval[1],.5), compute_mean = True, compute_sd = True)

# # pl.save_figure('pupil_amplitude_button-press.pdf', sub_folder = 'over_subs')

# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = response_pupil_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'anova')

# pl.save_figure('inc_pupil_response_button-press.pdf', sub_folder = 'over_subs/pupil/incorrect')

# # pl.open_figure(force=1)
# # pl.hline(y=0)
# # pl.event_related_pupil_difference(data = response_pupil_signals, conditions = ['PP','UP','PU','UU'], show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs),  4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,4.5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5))



# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = response_diff_signals, conditions = ['UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size (difference from predicted)', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'ttest')

# pl.save_figure('inc_pupil_difference_button-press.pdf', sub_folder = 'over_subs/pupil/incorrect')
# # pl.save_figure('pupil_amplitude-stimulus.pdf', sub_folder = 'over_subs/pupil')

# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = stimulus_pupil_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'anova')

# pl.save_figure('inc_pupil_response-stimulus.pdf', sub_folder = 'over_subs/pupil/incorrect')

# # pl.open_figure(force=1)
# # pl.hline(y=0)
# # pl.event_related_pupil_difference(data = stimulus_pupil_signals, conditions = ['PP','UP','PU','UU'], show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5))


# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = stimulus_diff_signals, conditions = ['UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size (difference from predicted)', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'ttest')

# pl.save_figure('inc_pupil_difference-stimulus.pdf', sub_folder = 'over_subs/pupil/incorrect')



# incorrect_response_pupil_signals = response_pupil_signals
# incorrect_stimulus_pupil_signals = stimulus_pupil_signals


keys = ['PP','UP','PU','UU']

keymap = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}

pl.open_figure(force=1)

for i in range(len(keys)):

	ax = pl.subplot(2,2,i+1)

	ax.set_title(keymap[keys[i]])

	pl.event_related_pupil_average(data = {'correct': correct_stimulus_pupil_signals[keys[i]], 'incorrect': incorrect_stimulus_pupil_signals[keys[i]]},
								   conditions = ['correct','incorrect'], compute_mean = True, compute_sd = True, with_stats = True, stats_type = 'ttest', x_lim = [5, 55], y_lim = [-0.25, 0.9], show_legend = True, xticks = np.arange(5,55,5), xticklabels = np.arange(-0.5, 5.5, 0.5), onset_marker = 10)

# pl.show_figure()

pl.save_figure('correct-v-incorrect_per_PE.pdf', sub_folder = 'over_subs/pupil')

# embed()

# f = plt.figure()

# plt.subplot(2,2,1)
# plt.title('No PE')
# plt.plot(np.mean(correct_stimulus_pupil_signals['PP'], axis=0), label='correct')
# plt.plot(np.mean(incorrect_stimulus_pupil_signals['PP'], axis=0), label='incorrect')

# plt.axis([10,55,-0.25,0.7])
# plt.legend()

# plt.subplot(2,2,2)
# plt.title('TR PE')
# plt.plot(np.mean(correct_stimulus_pupil_signals['UP'], axis=0), label='correct')
# plt.plot(np.mean(incorrect_stimulus_pupil_signals['UP'], axis=0), label='incorrect')

# plt.axis([10,55,-0.25,0.7])
# plt.legend()

# plt.subplot(2,2,3)
# plt.title('TI PE')
# plt.plot(np.mean(correct_stimulus_pupil_signals['PU'], axis=0), label='correct')
# plt.plot(np.mean(incorrect_stimulus_pupil_signals['PU'], axis=0), label='incorrect')

# plt.axis([10,55,-0.25,0.7])
# plt.legend()

# plt.subplot(2,2,4)
# plt.title('TR+TI PE')
# plt.plot(np.mean(correct_stimulus_pupil_signals['UU'], axis=0), label='correct')
# plt.plot(np.mean(incorrect_stimulus_pupil_signals['UU'], axis=0), label='incorrect')

# plt.axis([10,55,-0.25,0.7])
# plt.legend()

# plt.tight_layout()

# sn.despine()

# plt.savefig(os.path.join(figfolder,'over_subs','pupil','correct-with-incorrect.pdf'))

# data = pd.DataFrame()

# timepoint_data = []

# for t in range(np.diff(trial_deconvolution_interval) * deconv_sample_frequency):
# 	# tdata = pd.DataFrame()

# 	tdata = pd.DataFrame(np.hstack([np.array(correct_stimulus_diff_signals['UP'])[:,t],
# 								np.array(correct_stimulus_diff_signals['PU'])[:,t],
# 								np.array(correct_stimulus_diff_signals['UU'])[:,t]]),columns=['pupil'])	
# 	tdata['pe_type'] = np.hstack([['UP']*np.array(correct_stimulus_diff_signals['UP']).shape[0],
# 								  ['PU']*np.array(correct_stimulus_diff_signals['PU']).shape[0],
# 								  ['UU']*np.array(correct_stimulus_diff_signals['UU']).shape[0]])
# 	tdata['correct'] = np.ones(np.array(correct_stimulus_diff_signals['UP']).shape[0]+
# 					  		   np.array(correct_stimulus_diff_signals['PU']).shape[0]+
# 					           np.array(correct_stimulus_diff_signals['UU']).shape[0])

# 	tdata['subid'] = np.hstack([correct_stimulus_diff_subids['PU'],
# 								correct_stimulus_diff_subids['UP'],
# 								correct_stimulus_diff_subids['UU']])

# 	tdata_ic = pd.DataFrame(np.hstack([np.array(incorrect_stimulus_diff_signals['UP'])[:,t],
# 								np.array(incorrect_stimulus_diff_signals['PU'])[:,t],
# 								np.array(incorrect_stimulus_diff_signals['UU'])[:,t]]),columns=['pupil'])	
# 	tdata_ic['pe_type'] = np.hstack([['UP']*np.array(incorrect_stimulus_diff_signals['UP']).shape[0],
# 								  ['PU']*np.array(incorrect_stimulus_diff_signals['PU']).shape[0],
# 								  ['UU']*np.array(incorrect_stimulus_diff_signals['UU']).shape[0]])
# 	tdata_ic['correct'] = np.zeros(np.array(incorrect_stimulus_diff_signals['UP']).shape[0]+
# 					  		   np.array(incorrect_stimulus_diff_signals['PU']).shape[0]+
# 					           np.array(incorrect_stimulus_diff_signals['UU']).shape[0])

# 	tdata_ic['subid'] = np.hstack([incorrect_stimulus_diff_subids['PU'],
# 								incorrect_stimulus_diff_subids['UP'],
# 								incorrect_stimulus_diff_subids['UU']])	

# 	tdata = tdata.append(tdata_ic, ignore_index=True)

# 	tdata.to_hdf('pupil_task_data_diff.h5',key='t%i'%t,format='table',append=False,data_columns=['pupil','pe_type','correct','subid'])

# 	#timepoint_data.append(tdata.append(tdata_ic, ignore_index=True))


# data = pd.DataFrame()

# timepoint_data = []

# for t in range(np.diff(trial_deconvolution_interval) * deconv_sample_frequency):
# 	# tdata = pd.DataFrame()

# 	tdata = pd.DataFrame(np.hstack([np.array(correct_stimulus_pupil_signals['UP'])[:,t],
# 								np.array(correct_stimulus_pupil_signals['PU'])[:,t],
# 								np.array(correct_stimulus_pupil_signals['UU'])[:,t]]),columns=['pupil'])	
# 	tdata['pe_type'] = np.hstack([['UP']*np.array(correct_stimulus_pupil_signals['UP']).shape[0],
# 								  ['PU']*np.array(correct_stimulus_pupil_signals['PU']).shape[0],
# 								  ['UU']*np.array(correct_stimulus_pupil_signals['UU']).shape[0]])
# 	tdata['correct'] = np.ones(np.array(correct_stimulus_pupil_signals['UP']).shape[0]+
# 					  		   np.array(correct_stimulus_pupil_signals['PU']).shape[0]+
# 					           np.array(correct_stimulus_pupil_signals['UU']).shape[0])

# 	tdata['subid'] = np.hstack([correct_stimulus_diff_subids['PU'],
# 								correct_stimulus_diff_subids['UP'],
# 								correct_stimulus_diff_subids['UU']])

# 	tdata_ic = pd.DataFrame(np.hstack([np.array(incorrect_stimulus_pupil_signals['UP'])[:,t],
# 								np.array(incorrect_stimulus_pupil_signals['PU'])[:,t],
# 								np.array(incorrect_stimulus_pupil_signals['UU'])[:,t]]),columns=['pupil'])	
# 	tdata_ic['pe_type'] = np.hstack([['UP']*np.array(incorrect_stimulus_pupil_signals['UP']).shape[0],
# 								  ['PU']*np.array(incorrect_stimulus_pupil_signals['PU']).shape[0],
# 								  ['UU']*np.array(incorrect_stimulus_pupil_signals['UU']).shape[0]])
# 	tdata_ic['correct'] = np.zeros(np.array(incorrect_stimulus_pupil_signals['UP']).shape[0]+
# 					  		   np.array(incorrect_stimulus_pupil_signals['PU']).shape[0]+
# 					           np.array(incorrect_stimulus_pupil_signals['UU']).shape[0])

# 	tdata_ic['subid'] = np.hstack([incorrect_stimulus_diff_subids['PU'],
# 								incorrect_stimulus_diff_subids['UP'],
# 								incorrect_stimulus_diff_subids['UU']])	

# 	tdata = tdata.append(tdata_ic, ignore_index=True)

# 	tdata.to_hdf('pupil_task_data_nodiff.h5',key='t%i'%t,format='table',append=False,data_columns=['pupil','pe_type','correct','subid'])



# corr_incorr_diff_signals = {}

# for field in ['PP','UP','PU','UU']:
# 	corr_incorr_diff_signals[field] = np.array([i-c for c,i in zip(correct_stimulus_pupil_signals[field],incorrect_stimulus_pupil_signals[field])])



# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = corr_incorr_diff_signals, conditions=['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference (incorrect-correct)', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [None,None], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'ttest')
# pl.save_figure('correct_v_incorrect-stimulus.pdf', sub_folder = 'over_subs/pupil')

# corr_incorr_diff_signals = {}

# for field in ['PP','UP','PU','UU']:
# 	corr_incorr_diff_signals[field] = np.array([i-c for c,i in zip(correct_stimulus_diff_signals[field],incorrect_stimulus_diff_signals[field])])



# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = corr_incorr_diff_signals, conditions=['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','PU':'Task irrelevant','UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference (incorrect-correct)', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [None,None], compute_mean = True, compute_sd = True, with_stats = False, stats_type = 'ttest')
# pl.save_figure('correct_v_incorrect-difference-stimulus.pdf', sub_folder = 'over_subs/pupil')