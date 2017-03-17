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

sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ']#


low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 8
deconvolution_interval = np.array([0, 4.5])

down_fs = 100

pl = Plotter(figure_folder = figfolder)






#### PLOT AVERAGE PUPIL RESPONSE (pred v unpred)

# pl.open_figure(force=1)

# for subname in sublist:

# 	print subname
# 	# Organize filenames
# 	rawfolder = os.path.join(raw_data_folder,subname)
# 	sharedfolder = os.path.join(shared_data_folder,subname)
# 	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
# 	h5filename = os.path.join(sharedfolder,subname+'.h5')

# 	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 3, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)

# 	# Get pupil data (ev)
# 	pa.signal_per_trial()


# 	pl.subplot(2,4,sublist.index(subname)+1)

# 	#pupil_signals = {'pred': [], 'unpred': []}
# 	pupil_signals = {'pred': [],
# 					 'unpred': []}

# 	for key,trial_signal in pa.trial_signals.items():
# 		if key < 10:
# 			pupil_signals['pred'].extend(trial_signal[:,5:] - trial_signal[:,:5].mean())
# 		else:
# 			pupil_signals['unpred'].extend(trial_signal[:,5:] - trial_signal[:,:5].mean())

# 	pl.event_related_pupil_average(pupil_signal = pupil_signals, xticks = [0,10,20,30,40], xticklabels = [0,1,2,3,4], title = 'S%i'%sublist.index(subname), compute_mean = True, show_legend = False)

# 	pa.unload_data()


# pl.save_figure('all-ev_pupil_average.pdf')


#### PLOT AVERAGE PUPIL DIFFERENCE (pred - unpred)

# pl.open_figure(force=1)

# pl.subplot(1,2,1, title= 'Average pupil difference')

# pupil_signals = {'pred': [],
# 				 'unpred': []}

# for subname in sublist:

# 	print subname
# 	# Organize filenames
# 	rawfolder = os.path.join(raw_data_folder,subname)
# 	sharedfolder = os.path.join(shared_data_folder,subname)
# 	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
# 	h5filename = os.path.join(sharedfolder,subname+'.h5')

# 	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 3, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)

# 	# Get pupil data (ev)
# 	pa.signal_per_trial()

# 	for key,trial_signal in pa.trial_signals.items():
# 		if key < 10:
# 			pupil_signals['pred'].extend(trial_signal[:,5:] - trial_signal[:,:5].mean())
# 		else:
# 			pupil_signals['unpred'].extend(trial_signal[:,5:] - trial_signal[:,:5].mean())

# 	pa.unload_data()


# pl.event_related_pupil_difference(data = pupil_signals, conditions = ['pred','unpred'], show_legend = False, ylabel= 'Pupil size difference (sd, pred - unpred)', xlabel = 'Time after stimulus onset (s)')


#### PLOT AVERAGE PUPIL DIFFERENCE (pred, various unpreds)

# pl.open_figure()

# pl.subplot(1,2,2, title= 'Average pupil difference')

pupil_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}
power_signals = {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}

power_time_window = [10,25]

for subname in sublist:

	print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 3, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)

	# Get pupil data (ev)
	pa.signal_per_trial()

	for key,trial_signal in pa.trial_signals.items():
		if key < 10:
			pupil_signals['PP'].extend(trial_signal - trial_signal[:,:5].mean())
		elif key < 30:
			pupil_signals['UP'].extend(trial_signal - trial_signal[:,:5].mean())
		elif key < 50:
			pupil_signals['PU'].extend(trial_signal - trial_signal[:,:5].mean())
		else:
			pupil_signals['UU'].extend(trial_signal - trial_signal[:,:5].mean())

	psignals = [[],[],[],[]]

	all_signals = []

	for key,trial_signal in pa.trial_signals.items():
		trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]] - trial_signal[:,:5].mean()

		all_signals.extend(trial_signal)

	msignal = np.mean(all_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	for key,trial_signal in pa.trial_signals.items():
		trial_signal = trial_signal[:,power_time_window[0]:power_time_window[1]] - trial_signal[:,:5].mean()

		power_signal = np.dot(trial_signal, msignal)/msignal_norm

		if key < 10:
			power_signals['PP'].extend(power_signal)
		elif key < 30:
			power_signals['UP'].extend(power_signal)
		elif key < 50:
			power_signals['PU'].extend(power_signal)
		else:
			power_signals['UU'].extend(power_signal)

	pa.unload_data()

pl.open_figure(force=1)
pl.subplot(1,2,1, title='Average pupil difference')
pl.event_related_pupil_difference(data = pupil_signals, conditions = ['PP','UP','PU','UU'], show_legend = True, xticks = np.arange(0,40,5), xticklabels = np.arange(-5,3.5,.5), ylabel= 'Pupil size difference (sd, pred - unpred)', xlabel = 'Time after stimulus onset (s)')



pl.subplot(1,2,2, title='Average pupil amplitude')
pl.pupil_amplitude_per_condition(data = power_signals, conditions = ['PP','UP','PU','UU'], with_error = True)


#### PLOT AVERAGE PUPIL AMPLITUDE

pl.save_figure('all-ev_pupil_summary-things.pdf')