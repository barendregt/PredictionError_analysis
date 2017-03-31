import numpy as np
import scipy as sp

from math import *
import os,glob,sys

import cPickle as pickle
import pandas as pd

from IPython import embed

# sys.path.append('tools/')
from BehaviorAnalyzer import BehaviorAnalyzer
from Plotter import Plotter


raw_data_folder = '/home/barendregt/Projects/PredictionError/Psychophysics/Data' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/PredictionError/Figures'

#sublist = ['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM','AN']#
# sublist = ['AA','AB','AC','AD','AF','AG','AH','AI','AJ','AM']
# sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ']
sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AD','AE','AK','AL','AM','AN']
sbsetting = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 8
deconvolution_interval = np.array([-1.5, 4.5])

down_fs = 100

pl = Plotter(figure_folder = figfolder)


### PLOT AVERAGE PUPIL RESPONSE (pred v unpred)

pl.open_figure(force=1)

zero_point = 15

for subname in sublist:

	print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, verbosity =0, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)

	# pa.recombine_signal_blocks(reference_phase=7)

	# Get pupil data (ev)
	pa.signal_per_trial()


	pl.subplot(4,4,sublist.index(subname)+1, title='S%i'%sublist.index(subname))

	#pupil_signals = {'pred': [], 'unpred': []}
	pupil_signals = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}

	for key,trial_signal in pa.trial_signals.items():
		if key < 10:
			pupil_signals['PP'].extend(trial_signal - trial_signal[:,zero_point][:,np.newaxis])
		elif key < 30:
			pupil_signals['UP'].extend(trial_signal - trial_signal[:,zero_point][:,np.newaxis])
		elif key < 50:
			pupil_signals['PU'].extend(trial_signal - trial_signal[:,zero_point][:,np.newaxis])
		else:
			pupil_signals['UU'].extend(trial_signal - trial_signal[:,zero_point][:,np.newaxis])						

	pl.event_related_pupil_average(data = pupil_signals, xticks = np.arange(0,60,15), xticklabels = np.arange(-1.5,4.5,1.5), title = 'S%i'%sublist.index(subname), compute_mean = True, show_legend = False)

	pa.unload_data()


pl.save_figure('all-ev_pupil_average.pdf', sub_folder = 'individual')

pl.open_figure(force=1)

power_time_window = [30,50]
zero_point = 15

for subname in sublist:
	print subname
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs,verbosity =0, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)

	# Get pupil data (ev)
	pa.signal_per_trial(only_correct = True)	


	power_signals = {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}
	

	ref_signal = []

	for key,trial_signal in pa.trial_signals.items():
		trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]] - trial_signal[:,zero_point][:,np.newaxis]

		ref_signal.extend(trial_signal)

	msignal = np.mean(ref_signal, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)#**2xx

	for key,trial_signal in pa.trial_signals.items():
		trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]] - trial_signal[:,zero_point][:,np.newaxis]

		power_signal = np.dot(trial_signal, msignal)/msignal_norm

		if key < 10:
			pass
			# power_signals['PP'].extend(power_signal)
		elif key < 30:
			power_signals['UP'].extend(power_signal)
		elif key < 50:
			power_signals['PU'].extend(power_signal)
		else:
			power_signals['UU'].extend(power_signal)


	pl.subplot(4,4,sublist.index(subname)+1, title='S%i'%sublist.index(subname))
	pl.bar_plot(data = power_signals, conditions = ['UP','PU','UU'], with_error = True, ylabel = 'Pupil amplitude (a.u.)')
pl.save_figure('all_pupil-amplitude.pdf', sub_folder = 'individual')

pl.open_figure(force=1)

for subname in sublist:
	print subname
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs,verbosity =0, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)


	sub_ie_scores = pa.compute_inverse_efficiency_scores()

	ie_scores = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}

	ie_scores['PP'].extend(sub_ie_scores[0])
	ie_scores['PP'].extend(sub_ie_scores[1])

	ie_scores['UP'].extend(sub_ie_scores[10] / np.median(sub_ie_scores[0]))
	ie_scores['UP'].extend(sub_ie_scores[20] / np.median(sub_ie_scores[1]))
	ie_scores['PU'].extend(sub_ie_scores[30] / np.median(sub_ie_scores[0]))
	ie_scores['PU'].extend(sub_ie_scores[40] / np.median(sub_ie_scores[1]))
	ie_scores['UU'].extend(sub_ie_scores[50] / np.median(sub_ie_scores[0]))
	ie_scores['UU'].extend(sub_ie_scores[60] / np.median(sub_ie_scores[1]))

	for (key,item) in ie_scores.items():
		ie_scores[key] = np.array(ie_scores[key]) * np.median(ie_scores['PP'])

	pl.subplot(4,4,sublist.index(subname)+1, title='S%i'%sublist.index(subname))
	pl.bar_plot(data = ie_scores, conditions = ['UP','PU','UU'], ylabel='', with_error = True, x_lim = [0.5, None], y_lim = [None, None])

	# pa.unload_data()

pl.save_figure('all_inverse-efficiency.pdf', sub_folder = 'individual')