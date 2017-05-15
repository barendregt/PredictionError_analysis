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
sublist = ['AA','AB','AC','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','DA','DB','DC','DD','DE','DF']#
# sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AD','AE','AK','AL','AM','AN']
sbsetting = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

low_pass_pupil_f, high_pass_pupil_f = 4.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
response_deconvolution_interval = np.array([-1, 4.5])
stimulus_deconvolution_interval = np.array([-0.5, 4.5])

down_fs = 50

pl = Plotter(figure_folder = figfolder)



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

power_time_window = [int(0.5*(signal_sample_frequency/down_fs)),int(5*(signal_sample_frequency/down_fs))]#[15,30]
zero_point = 15

# all_ie_scores = []
all_rts = []

for subname in sublist:

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, sort_by_date = sbsetting[sublist.index(subname)], signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

	# redo signal extraction
	# pa.recombine_signal_blocks(force_rebuild = True)

	#sub_rts = pa.compute_reaction_times()

	# # Get pupil data (ev)
	pa.signal_per_trial(only_correct = True, reference_phase = 7, with_rt = True, baseline_type = 'relative', baseline_period = [-0.5, 0.0], force_rebuild=False)

	pa.get_IRF()

	all_sub_IRF['button_press'].append(pa.sub_IRF['button_press'])


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
				response_pupil_signals['PP'].append(np.mean(trial_signal, axis=0))
			elif key < 30:
				response_pupil_signals['UP'].append(np.mean(trial_signal, axis=0))
			elif key < 50:
				response_pupil_signals['PU'].append(np.mean(trial_signal, axis=0))
			else:
				response_pupil_signals['UU'].append(np.mean(trial_signal, axis=0))


			if key < 10:
				response_diff_signals['PP'].extend((trial_signal*msignal)/msignal_norm)
			elif key < 30:
				response_diff_signals['UP'].extend((trial_signal*msignal)/msignal_norm)
			elif key <50:
				response_diff_signals['PU'].extend((trial_signal*msignal)/msignal_norm)
			else:
				response_diff_signals['UU'].extend((trial_signal*msignal)/msignal_norm)
	except:
		embed()

	ref_signals = []

	for key,trial_signal in pa.trial_signals.items():
		if key < 10:
			trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]]# - trial_signal[:,zero_point][:,np.newaxis]

			ref_signals.extend(trial_signal)

	msignal = np.mean(ref_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	for key,trial_signal in pa.trial_signals.items():
		trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]]# - trial_signal[:,zero_point][:,np.newaxis]

		power_signal = np.dot(trial_signal, msignal)/msignal_norm

		if key < 10:
			pass
			# power_signals['PP'].extend(power_signal)
		if key < 30:
			power_signals['UP'].extend(power_signal)
		elif key < 50:
			power_signals['UU'].extend(power_signal)
		else:
			power_signals['PU'].extend(power_signal)

	pa.deconvolution_interval = stimulus_deconvolution_interval
	pa.signal_per_trial(only_correct = True, reference_phase = 4, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False)

	pa.get_IRF()

	all_sub_IRF['stimulus'].append(pa.sub_IRF['stimulus'])

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
		elif key < 30:
			stimulus_pupil_signals['UP'].extend(trial_signal)
		elif key < 50:
			stimulus_pupil_signals['PU'].extend(trial_signal)
		else:
			stimulus_pupil_signals['UU'].extend(trial_signal)


		if key < 10:
			stimulus_diff_signals['PP'].extend((trial_signal*msignal)/msignal_norm)
		elif key < 30:
			stimulus_diff_signals['UP'].extend((trial_signal*msignal)/msignal_norm)
		elif key <50:
			stimulus_diff_signals['PU'].extend((trial_signal*msignal)/msignal_norm)
		else:
			stimulus_diff_signals['UU'].extend((trial_signal*msignal)/msignal_norm)

	sub_rts = pa.compute_reaction_times()

	all_rts.append(sub_rts[0])
	all_rts.append(sub_rts[1])

	rts['UP'].append(np.mean(sub_rts[10] / np.median(sub_rts[0])))
	rts['UP'].append(np.mean(sub_rts[20] / np.median(sub_rts[1])))
	rts['PU'].append(np.mean(sub_rts[30] / np.median(sub_rts[0])))
	rts['PU'].append(np.mean(sub_rts[40] / np.median(sub_rts[1])))
	rts['UU'].append(np.mean(sub_rts[50] / np.median(sub_rts[0])))
	rts['UU'].append(np.mean(sub_rts[60] / np.median(sub_rts[1])))


	sub_pc = pa.compute_percent_correct()

	pc['PP'].append(sub_pc[0])
	pc['PP'].append(sub_pc[1])
	pc['UP'].append(sub_pc[10])
	pc['UP'].append(sub_pc[20])
	pc['PU'].append(sub_pc[30])
	pc['PU'].append(sub_pc[40])
	pc['UU'].append(sub_pc[50])
	pc['UU'].append(sub_pc[60])

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

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_diff_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','UU':'Task irrelevant','PU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0, 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,4.5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5), compute_mean = True, compute_sd = True)

pl.save_figure('pupil_amplitude_button-press.pdf', sub_folder = 'over_subs')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_pupil_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','UU':'Task irrelevant','PU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0, 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,4.5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5), compute_mean = True, compute_sd = True)

pl.save_figure('pupil_response_button-press.pdf', sub_folder = 'over_subs')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_difference(data = response_pupil_signals, conditions = ['PP','UP','PU','UU'], show_legend=True, ylabel = 'Pupil size', x_lim = [0, 45], xticks = np.arange(0,4.5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5))

pl.save_figure('pupil_difference_button-press.pdf', sub_folder = 'over_subs')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_diff_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','UU':'Task irrelevant','PU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0, 5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), compute_mean = True, compute_sd = True)

pl.save_figure('pupil_amplitude-stimulus.pdf', sub_folder = 'over_subs')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_pupil_signals, conditions = ['PP','UP','PU','UU'], signal_labels = {'PP': 'Predicted', 'UP': 'Task relevant','UU':'Task irrelevant','PU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0, 5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), compute_mean = True, compute_sd = True)

pl.save_figure('pupil_response-stimulus.pdf', sub_folder = 'over_subs')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_difference(data = stimulus_pupil_signals, conditions = ['PP','UP','PU','UU'], show_legend=True, ylabel = 'Pupil size', x_lim = [0, 5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5))

pl.save_figure('pupil_difference-stimulus.pdf', sub_folder = 'over_subs')


pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = all_sub_IRF, conditions = ['stimulus','button_press'], show_legend=True, ylabel = 'Pupil size', compute_mean = True, compute_sd = True)

pl.save_figure('pupil_IRF.pdf', sub_folder = 'over_subs')


# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_average(data = pupil_signals, conditions = ['PP','UP','PU','UU'], show_legend=True, x_lim = [0, 45], xticks = np.arange(0,45,5), xticklabels = np.arange(deconvolution_interval[0]+0.5, deconvolution_interval[1]+0.5,.5), compute_mean = True)

# pl.save_figure('pupil_average.pdf', sub_folder = 'over_subs')

# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.event_related_pupil_difference(data = diff_signals, conditions = ['PP','UP','PU','UU'], show_legend=True, x_lim = [0, 45], xticks = np.arange(0,45,5), xticklabels = np.arange(deconvolution_interval[0]+0.5, deconvolution_interval[1]+0.5,.5), with_error = False)

# pl.save_figure('pupil_difference.pdf', sub_folder = 'over_subs')

# pl.open_figure(force=1)
# # pl.figure.suptitle('Reaction time')
# pl.hatline(x = (2.5,3.5), y = (np.mean(rts['UP'])+np.mean(rts['PU']),np.mean(rts['UP'])+np.mean(rts['PU'])))
# pl.bar_plot(data = rts, conditions = ['UP','PU','UU'], ylabel='Relative reaction time (% of predicted)', with_error = True, x_lim = [0.5, None],xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error', y_lim = [1.0, 1.21], yticks = np.arange(1.0,1.4,.05), yticklabels = [str(val)+"%" for val in np.arange(100,140,5)])

# pl.save_figure('reaction_times.pdf', sub_folder = 'over_subs')

# # # rtdata = pd.DataFrame(data = np.vstack([np.hstack([rts['UP'], rts['PU'], rts['UU']]), np.hstack([['PU']*len(rts['PU']), ['UP']*len(rts['UP']), ['UU']*len(rts['UU'])])]).T, columns = ['RT','PE_type'])
# # # rtdata.to_csv('rt.csv')

# pl.open_figure(force=1)
# # pl.figure.suptitle('Performance')
# pl.bar_plot(data = pc, conditions = ['PP','UP','PU','UU'],xticklabels = ['None','Task relevant','Task irrelevant','Both'],xlabel='Prediction error', ylabel='Performance (% correct)', with_error = True, x_lim = [0.5, None], y_lim = [0.5, 1.0], yticks = np.arange(0.0,1.1,.1), yticklabels = np.arange(0,110,10))

# pl.save_figure('percent_correct.pdf', sub_folder = 'over_subs')
pl.open_figure(force=1)
# pl.figure.suptitle('Pupil amplitude')
# pl.subplot(1,2,1, title='Pupil amplitude')
pl.bar_plot(data = power_signals, conditions = ['UP','PU','UU'], with_error = True, ylabel = 'Pupil amplitude (a.u.)', x_lim = [0.5, None], y_lim = [0.6, 1.21], yticks = np.arange(0.6, 2.0, 0.2), yticklabels = np.arange(0.8,2.2,0.2), xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error')

pl.save_figure('pupil_amplitude_bar.pdf', sub_folder = 'over_subs')

pl.open_figure(force=1)
# pl.figure.suptitle('Reaction time')
# pl.hatline(x = (2.5,3.5), y = (np.mean(rts['UP'])+np.mean(rts['PU']),np.mean(rts['UP'])+np.mean(rts['PU'])))
pl.bar_plot(data = rts, conditions = ['UP','PU','UU'], ylabel='Relative reaction time (% of predicted)', with_error = True, with_data_points = True, x_lim = [0.5, None],xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error', y_lim = [1.0, 1.21], yticks = np.arange(1.0,1.4,.05), yticklabels = [str(val)+"%" for val in np.arange(100,140,5)])

pl.save_figure('reaction_times.pdf', sub_folder = 'over_subs')

# # rtdata = pd.DataFrame(data = np.vstack([np.hstack([rts['UP'], rts['PU'], rts['UU']]), np.hstack([['PU']*len(rts['PU']), ['UP']*len(rts['UP']), ['UU']*len(rts['UU'])])]).T, columns = ['RT','PE_type'])
# # rtdata.to_csv('rt.csv')

pl.open_figure(force=1)
# pl.figure.suptitle('Performance')
pl.bar_plot(data = pc, conditions = ['PP','UP','PU','UU'],xticklabels = ['None','Task relevant','Task irrelevant','Both'],xlabel='Prediction error', ylabel='Performance (% correct)', with_error = True, x_lim = [0.5, None], y_lim = [0.5, 1.0], yticks = np.arange(0.0,1.1,.1), yticklabels = np.arange(0,110,10))

pl.save_figure('percent_correct.pdf', sub_folder = 'over_subs')


# # # pl.save_figure('all-ev_pupil_summary-things.pdf', sub_folder = 'summary')