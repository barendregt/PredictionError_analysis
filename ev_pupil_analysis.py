import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

from math import *
import os,glob,sys

import cPickle as pickle
import pandas as pd

from IPython import embed

# sys.path.append('tools/')
from BehaviorAnalyzer import BehaviorAnalyzer


raw_data_folder = '/home/barendregt/Projects/Attention_Prediction/Psychophysics/Data' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/PredictionError/Figures'

sublist = ['s1','s2','s4','s5','s6']#['s1','s2','s3','s4','s5','s6']#['s1','s2','s4']['s1','s2',[

# subname = 's1'#'tk2'#'s3'#'mb2'#'tk2'#'s3'#'tk2'

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 8
deconvolution_interval = np.array([-0.5, 6.5])

down_fs = 100

sn.set(style='ticks')

all_sub_rts = [[],[],[]]
 
def run_analysis(subname):	

	print '[main] Running analysis for %s' % (subname)

	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)

	csvfilename = glob.glob(rawfolder + '/*.csv')[-1]

	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)

	# pa.recombine_signal_blocks()
	# Pupil data

	pa.signal_per_trial()

	plt.figure()
	# EV signals

	plt.subplot(2,2,1)
	plt.title('EV pupil response')
	#plt.figure(figsize=(10,8))

	pred_signal = []
	unpred_signal = []

	for key,trial_signal in pa.trial_signals.items():
		if key < 10:
			pred_signal.extend(trial_signal[:,5:] - trial_signal[:,:5].mean())
		else:
			unpred_signal.extend(trial_signal[:,5:] - trial_signal[:,:5].mean())

	
	plt.plot(np.mean(pred_signal, axis=0), label='expected')
	plt.plot(np.mean(unpred_signal, axis=0), label='unexpected')

	plt.ylabel('Pupil response (perc. change)')
	plt.xlabel('Time (s)')

	plt.xticks(np.arange(0,70,10), np.arange(7))

	plt.legend()

	# Amplitude per condition
	plt.subplot(2,2,2)
	plt.title('Pupil amplitude')

	peak_window = [30,50]
	psignals = [[],[],[],[]]

	for key,trial_signal in pa.trial_signals.items():
		trial_signal = trial_signal[:,5:] - trial_signal[:,:5].mean()

		msignal = trial_signal.mean(axis=0)

		power_signal = np.array([(signal*msignal)/(np.linalg.norm(msignal, ord=2)**2) for signal in trial_signal])

		if key < 10:
			psignals[0].extend(power_signal[:,peak_window[0]:peak_window[1]].max(axis=1))
		elif key < 30:
			psignals[1].extend(power_signal[:,peak_window[0]:peak_window[1]].max(axis=1))
		elif key < 50:
			psignals[2].extend(power_signal[:,peak_window[0]:peak_window[1]].max(axis=1))
		else:
			psignals[3].extend(power_signal[:,peak_window[0]:peak_window[1]].max(axis=1))


	psignals_mean = [np.mean(s) for s in psignals]
	psignals_std  = [np.std(s)/np.sqrt(len(s)) for s in psignals]
	plt.bar(range(1,5), psignals_mean)
	plt.errorbar(np.arange(1.5,5.5), psignals_mean, psignals_std, fmt = '.')

	plt.xticks(range(0,5), ['','P','P-U','U-P','U-U'])

	# REACTION TIME
	plt.subplot(2,2,3)
	rts = [[],[],[],[]]

	trial_parameters = pa.read_trial_data(pa.combined_h5_filename)

	trial_color_values = np.unique(abs(trial_parameters['trial_color']))
	trial_ori_values = np.unique(abs(trial_parameters['trial_orientation']))

	trial_limits = [[trial_color_values[1], trial_color_values[-2]],
		 			[trial_ori_values[1], trial_ori_values[-2]]]

	for tcode in np.unique(trial_parameters['trial_codes']):
		if tcode%2==0:
			selected_rts = trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) & (trial_parameters['trial_codes']==tcode) & ((trial_parameters['trial_color'] > trial_limits[0][0]) & (trial_parameters['trial_color'] < trial_limits[0][1]))]
		else:
			selected_rts = trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) & (trial_parameters['trial_codes']==tcode) & ((trial_parameters['trial_orientation'] > trial_limits[1][0]) & (trial_parameters['trial_orientation'] < trial_limits[1][1]))]

		if tcode < 10:
			rts[0].extend(selected_rts)
		elif tcode < 30:
			rts[1].extend(selected_rts)
		elif tcode < 50:
			rts[2].extend(selected_rts)
		else:
			rts[3].extend(selected_rts)

	rts_mean = [np.mean(s) for s in rts]
	rts_std  = [np.std(s)/np.sqrt(len(s)) for s in rts]
	rts_mean[1:] /= rts_mean[0]
	rts_std[1:] /= rts_mean[0]
	# plt.bar(range(1,4), rts_mean[1:])
	plt.errorbar(np.arange(1.5,4.5), rts_mean[1:], rts_std[1:], fmt = '.')

	plt.xticks(range(1,4), ['P-U','U-P','U-U'])

	all_sub_rts[0].append(rts_mean[1]/rts_mean[0])
	all_sub_rts[1].append(rts_mean[2]/rts_mean[0])
	all_sub_rts[2].append(rts_mean[3]/rts_mean[0])

	# plt.show()
	plt.savefig(os.path.join(figfolder,subname + '-ev_pupil.png'))

	# embed()	

	pa.unload_data()

# run_analysis('s2')
[run_analysis(sub) for sub in sublist]

plt.figure()

plt.bar([1,2,3],np.mean(all_sub_rts, axis=1))
plt.plot([1,2,3],all_sub_rts,'o')

plt.savefig(os.path.join(figfolder,'all_sub_rts.png'))