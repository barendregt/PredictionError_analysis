from __future__ import division

import numpy as np
import scipy as sp

from numpy import *
import scipy as sp
from pandas import *

from joblib import Parallel, delayed
import multiprocessing

from math import *
import os,glob,sys

import cPickle as pickle
import pandas as pd

from IPython import embed

# sys.path.append('tools/')
from BehaviorAnalyzer import BehaviorAnalyzer
from Plotter import Plotter


raw_data_folder = '/home/raw_data/2017/visual/PredictionError/Behavioural/Reaction_times'#'/home/barendregt/Projects/PredictionError/Psychophysics/Data' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/PredictionError/Figures'

#sublist = ['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN']#
# sublist = ['AA','AB','AC','AD','AF','AG','AH','AI','AJ','AM']
sublist = ['AA','AB','AC','AE','AF','AG','AH','AI','AJ']
# sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AD','AE','AK','AL','AM','AN']
sbsetting = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

low_pass_pupil_f, high_pass_pupil_f = 4.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
deconvolution_interval = np.array([-1.5, 4.5])

down_fs = 100

# pl = Plotter(figure_folder = figfolder)



#### PLOT AVERAGES OVER SUBS

# pl.open_figure()

# # pl.subplot(1,2,2, title= 'Average pupil difference')

# pupil_signals = {'PP': [],
# 				 'UP': [],
# 				 'PU': [],
# 				 'UU': []}
# diff_signals  = {'PP': [],
# 				 'UP': [],
# 				 'PU': [],
# 				 'UU': []}				 
# power_signals = {'PP': [],
# 				 'UP': [],
# 				 'PU': [],
# 				 'UU': []}
# ie_scores 	  = {'PP': [],
# 				 'UP': [],
# 				 'PU': [],
# 				 'UU': []}
# rts 	  	  = {'PP': [],
# 				 'UP': [],
# 				 'PU': [],
# 				 'UU': []}
# pc 	  	  	  = {'PP': [],
# 				 'UP': [],
# 				 'PU': [],
# 				 'UU': []}			
# all_ie_scores = {'PP': [],
# 				 'UP': [],
# 				 'PU': [],
# 				 'UU': []}	
# power_time_window = [30,50]#[15,30]
# zero_point = 15

# # all_ie_scores = []
# all_rts = []

def run_analysis(subname):

	print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, low_pass_pupil_fr = low_pass_pupil_f, signal_downsample_factor = down_fs, sort_by_date = sbsetting[sublist.index(subname)], signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval, verbosity = 1)

	# redo signal extraction
	pa.recombine_signal_blocks(force_rebuild = True)





[run_analysis(subname) for subname in sublist]
# num_cores = multiprocessing.cpu_count()

# Parallel(n_jobs=num_cores)(delayed(run_analysis)(subname) for subname in sublist)

	# # Get pupil data (ev)
	# pa.signal_per_trial(only_correct = True)

	# ref_signals = []

	# for key,trial_signal in pa.trial_signals.items():
	# 	if key < 10:
	# 		#trial_signal = trial_signal - trial_signal[:,zero_point][:,np.newaxis]

	# 		ref_signals.extend(trial_signal)

	# msignal = np.mean(ref_signals, axis=0)
	# msignal_norm = np.linalg.norm(msignal, ord=2)#**2

	# for key, trial_signal in pa.trial_signals.items():

	# 	#trial_signal -= trial_signal[:,:zero_point].mean()
	# 	#trial_signal[:,zero_point] = 0

	# 	if key < 10:
	# 		diff_signals['PP'].extend((trial_signal*msignal))#/msignal_norm)
	# 	elif key < 30:
	# 		diff_signals['PU'].extend((trial_signal*msignal))#/msignal_norm)
	# 	elif key <50:
	# 		diff_signals['UP'].extend((trial_signal*msignal))#/msignal_norm)
	# 	else:
	# 		diff_signals['UU'].extend((trial_signal*msignal))#/msignal_norm)

	# ref_signals = []

	# for key,trial_signal in pa.trial_signals.items():
	# 	if key < 10:
	# 		trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]] - trial_signal[:,zero_point][:,np.newaxis]

	# 		ref_signals.extend(trial_signal)

	# msignal = np.mean(ref_signals, axis=0)
	# msignal_norm = np.linalg.norm(msignal, ord=2)#**2

	# for key,trial_signal in pa.trial_signals.items():
	# 	trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]] - trial_signal[:,zero_point][:,np.newaxis]

	# 	power_signal = np.dot(trial_signal, msignal)/msignal_norm

	# 	if key < 10:
	# 		pass
	# 		# power_signals['PP'].extend(power_signal)
	# 	if key < 30:
	# 		power_signals['UP'].extend(power_signal)
	# 	elif key < 50:
	# 		power_signals['PU'].extend(power_signal)
	# 	else:
	# 		power_signals['UU'].extend(power_signal)

	# sub_ie_scores = pa.compute_inverse_efficiency_scores()

	# all_ie_scores['PP'].extend(sub_ie_scores[0])
	# all_ie_scores['PP'].extend(sub_ie_scores[1])
	# all_ie_scores['UP'].extend(sub_ie_scores[10])
	# all_ie_scores['UP'].extend(sub_ie_scores[20])
	# all_ie_scores['PU'].extend(sub_ie_scores[30])
	# all_ie_scores['PU'].extend(sub_ie_scores[40])
	# all_ie_scores['UU'].extend(sub_ie_scores[50])
	# all_ie_scores['UU'].extend(sub_ie_scores[60])

	# ie_scores['PP'].extend(sub_ie_scores[10] / np.median(sub_ie_scores[0]))
	# ie_scores['UP'].extend(sub_ie_scores[10] / np.median(sub_ie_scores[0]))
	# ie_scores['UP'].extend(sub_ie_scores[20] / np.median(sub_ie_scores[1]))
	# ie_scores['PU'].extend(sub_ie_scores[30] / np.median(sub_ie_scores[0]))
	# ie_scores['PU'].extend(sub_ie_scores[40] / np.median(sub_ie_scores[1]))
	# ie_scores['UU'].extend(sub_ie_scores[50] / np.median(sub_ie_scores[0]))
	# ie_scores['UU'].extend(sub_ie_scores[60] / np.median(sub_ie_scores[1]))


	# sub_rts = pa.compute_reaction_times()

	# all_rts.extend(sub_rts[0])
	# all_rts.extend(sub_rts[1])

	# rts['UP'].extend(sub_rts[10] / np.median(sub_rts[0]))
	# rts['UP'].extend(sub_rts[20] / np.median(sub_rts[1]))
	# rts['PU'].extend(sub_rts[30] / np.median(sub_rts[0]))
	# rts['PU'].extend(sub_rts[40] / np.median(sub_rts[1]))
	# rts['UU'].extend(sub_rts[50] / np.median(sub_rts[0]))
	# rts['UU'].extend(sub_rts[60] / np.median(sub_rts[1]))


	# sub_pc = pa.compute_percent_correct()

	# pc['PP'].append(sub_pc[0])
	# pc['PP'].append(sub_pc[1])
	# pc['UP'].append(sub_pc[10])
	# pc['UP'].append(sub_pc[20])
	# pc['PU'].append(sub_pc[30])
	# pc['PU'].append(sub_pc[40])
	# pc['UU'].append(sub_pc[50])
	# pc['UU'].append(sub_pc[60])


# 	# pa.unload_data()


# embed()

# pl.open_figure(force=1)
# # pl.figure.suptitle('Pupil amplitude')
# # pl.subplot(1,2,1, title='Pupil amplitude')
# pl.bar_plot(data = power_signals, conditions = ['UP','PU','UU'], with_error = True, ylabel = 'Pupil amplitude (a.u.)', x_lim = [0.5, None], y_lim = [0.6, None], xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error category')

# pl.save_figure('pupil_amplitude.pdf', sub_folder = 'across_subs')

# # Convert IE scores back to milliseconds
# for (key,item) in ie_scores.items():
# 	ie_scores[key] = np.array(ie_scores[key]) * np.median(all_ie_scores['PP'])
# 	# rts[key] = np.array(rts[key]) * np.median(all_rts)

# # pl.subplot(1,2,2, title='Reaction times')
# # pl.open_figure(force=1)
# # # pl.figure.suptitle('Inverse Efficiency')
# # pl.hline(y = np.median(all_ie_scores['PP']), label = 'Predicted')
# # pl.bar_plot(data = ie_scores, conditions = ['UP','PU','UU'], ylabel='Corrected reaction time (ms)', with_error = True, x_lim = [0.5, None], xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error category', y_lim = [1.4, None], yticks = np.arange(0.0,1.8,.1), yticklabels = np.arange(0,1800,100))


# # pl.open_figure(force=1)
# # pl.hline(y=0)
# # pl.event_related_pupil_average(data = diff_signals, conditions = ['PP','UP','PU'], signal_labels = {'PP': 'Task relevant', 'UP': 'Task irrelevant','PU': 'Both'}, show_legend=True, x_lim = [0, 40], xticks = np.arange(0,40,5), xticklabels = np.arange(-.5,3.5,.5), compute_mean = True)

# # pl.save_figure('pupil_average.pdf', sub_folder = 'across_subs')

# pl.open_figure(force=1)
# # pl.figure.suptitle('Reaction time')
# # pl.hline(y = np.median(all_rts), label = 'Predicted')
# pl.bar_plot(data = rts, conditions = ['UP','PU','UU'], ylabel='Relative reaction time (% of predicted)', with_error = True, x_lim = [0.5, None],xticklabels = ['Task relevant','Task irrelevant','Both'], xlabel = 'Prediction error type', y_lim = [1.0, 1.2], yticks = np.arange(1.0,1.4,.05), yticklabels = [str(val)+"%" for val in np.arange(100,140,5)])

# pl.save_figure('reaction_times.pdf', sub_folder = 'across_subs')

# # rtdata = pd.DataFrame(data = np.vstack([np.hstack([rts['UP'], rts['PU'], rts['UU']]), np.hstack([['PU']*len(rts['PU']), ['UP']*len(rts['UP']), ['UU']*len(rts['UU'])])]).T, columns = ['RT','PE_type'])
# # rtdata.to_csv('rt.csv')

# pl.open_figure(force=1)
# pl.figure.suptitle('Performance')
# pl.bar_plot(data = pc, conditions = ['PP','UP','PU','UU'], ylabel='Performance (% correct)', with_error = True, x_lim = [0.5, None], y_lim = [0.5, 1.0], yticks = np.arange(0.0,1.1,.1), yticklabels = np.arange(0,110,10))

# pl.save_figure('percent_correct.pdf', sub_folder = 'across_subs')



# # pl.save_figure('all-ev_pupil_summary-things.pdf', sub_folder = 'summary')