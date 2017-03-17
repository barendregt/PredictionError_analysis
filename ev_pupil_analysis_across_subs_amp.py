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

sublist = ['AA','AB','AC','AF','AG']#
# sublist = ['s1','s2','s3','s4','s5','s6']#['s1','s2','s3','s4','s5','s6']#['s1','s2','s4']['s1','s2',[

# subname = 's1'#'tk2'#'s3'#'mb2'#'tk2'#'s3'#'tk2'

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 8
deconvolution_interval = np.array([0, 4.5])

down_fs = 100

sn.set(style='ticks')

all_sub_rts = [[],[],[]]
 
# def run_analysis(subname):	
plt.figure()
# psignals_mean = []
# all_rts_mean = []
for subname in sublist:

	print '[main] Running analysis for %s' % (subname)

	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)

	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]

	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 3, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)

	# pa.recombine_signal_blocks()
	# Pupil data
	pa.load_data()

	pa.signal_per_trial()

	# plt.figure()
	# EV signals
	# embed()
	plt.subplot(2,3,sublist.index(subname)+1)
	plt.title('S%i'%sublist.index(subname))
	#plt.figure(figsize=(10,8))

	# pred_signal = []
	# unpred_signal = []

	# for key,trial_signal in pa.trial_signals.items():
	# 	if key < 10:
	# 		pred_signal.extend(trial_signal - trial_signal[:,:5].mean())
	# 	else:
	# 		unpred_signal.extend(trial_signal - trial_signal[:,:5].mean())

	# plt.vlines(5, -np.Inf, np.Inf, linewidth=1.5, color='k')

	# plt.plot(np.mean(pred_signal, axis=0), label='expected')
	# plt.plot(np.mean(unpred_signal, axis=0), label='unexpected')

	# plt.xticks()

	# plt.ylabel('Pupil response (z)')
	# plt.xlabel('Time after stim onset (s)')

	# plt.xticks(np.arange(0,45,5).tolist())#, labels=np.arange(-0.5,4.0,5).tolist())

	# plt.legend()

	# sn.despine(offset=5)

	# Amplitude per condition
	# plt.subplot(len(sublist),2,2*sublist.index(subname)+2)
	# plt.title('Pupil amplitude')

	peak_window = [5,35]
	psignals = [[],[],[],[]]

	all_signals = []

	for key,trial_signal in pa.trial_signals.items():
		trial_signal = trial_signal[:,peak_window[0]:peak_window[1]] - trial_signal[:,:5].mean()

		all_signals.extend(trial_signal)

	msignal = np.mean(all_signals, axis=0)

	for key,trial_signal in pa.trial_signals.items():
		trial_signal = trial_signal[:,peak_window[0]:peak_window[1]] - trial_signal[:,:5].mean()
		# power_signal = np.array([np.dot(signal,msignal)/(np.linalg.norm(msignal, ord=2)**2) for signal in trial_signal])
		power_signal = np.dot(trial_signal, msignal)/(np.linalg.norm(msignal, ord=2)**2)

		if key < 10:
			# psignals[0].extend(power_signal[:,peak_window[0]:peak_window[1]].max(axis=1))
			psignals[0].extend(power_signal)
		elif key < 30:
			# psignals[1].extend(power_signal[:,peak_window[0]:peak_window[1]].max(axis=1))
			psignals[1].extend(power_signal)
		elif key < 50:
			# psignals[2].extend(power_signal[:,peak_window[0]:peak_window[1]].max(axis=1))
			psignals[2].extend(power_signal)
		else:
			# psignals[3].extend(power_signal[:,peak_window[0]:peak_window[1]].max(axis=1))
			psignals[3].extend(power_signal)


	psignals_mean = [np.mean(s) for s in psignals]
	psignals_std  = [np.std(s)/sqrt(len(s)) for s in psignals]

	plt.bar(np.arange(4), psignals_mean, yerr = psignals_std, color='w')

	plt.axis('tight')
	plt.xticks(np.arange(4)+0.25, ('P','P-U','U-P','U-U'))

	plt.tight_layout()

	# rts = [[],[],[],[]]

	# trial_parameters = pa.read_trial_data(pa.combined_h5_filename)

	# # trial_color_values = np.unique(abs(trial_parameters['trial_color']))
	# # trial_ori_values = np.unique(abs(trial_parameters['trial_orientation']))

	# # trial_limits = [[trial_color_values[1], trial_color_values[-2]],
	# # 	 			[trial_ori_values[1], trial_ori_values[-2]]]

	# # trial_pcs = np.zeros((trial_parameters.shape[0],1))

	# # for (tc,to) in zip(trial_color_values, trial_ori_values):
	# # 	# print (tc,to)
	# # 	trial_iis = np.array(((abs(trial_parameters['trial_color'])==tc) & (trial_parameters['task']==1)) | ((abs(trial_parameters['trial_orientation'])==to) & (trial_parameters['task']==2)),dtype=bool)
	# # 	trial_pcs[trial_iis] = trial_parameters['correct_answer'][trial_iis].mean()

	# # trial_pcs = trial_pcs.squeeze()

	# for tcode in np.unique(trial_parameters['trial_codes']):
	# 	# if tcode%2==0:
	# 	# 	selected_rts = trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) & (trial_parameters['trial_codes']==tcode) & ((trial_parameters['trial_color'] > trial_limits[0][0]) & (trial_parameters['trial_color'] < trial_limits[0][1]))]
	# 	# else:
	# 	# 	selected_rts = trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) & (trial_parameters['trial_codes']==tcode) & ((trial_parameters['trial_orientation'] > trial_limits[1][0]) & (trial_parameters['trial_orientation'] < trial_limits[1][1]))]
	# 	tc_iis = np.array((trial_parameters['correct_answer']==1) & (trial_parameters['trial_codes']==tcode), dtype=bool)
	# 	selected_rts = trial_parameters['reaction_time'][tc_iis]# / trial_pcs[tc_iis]

	# 	if tcode < 10:
	# 		rts[0].extend(selected_rts)
	# 	elif tcode < 30:
	# 		rts[1].extend(selected_rts)
	# 	elif tcode < 50:
	# 		rts[2].extend(selected_rts)
	# 	else:
	# 		rts[3].extend(selected_rts)

	# rts_mean = [np.mean(s) for s in rts]
	# # rts_std  = [np.std(s)/np.sqrt(len(s)) for s in rts]
	# rts_mean /= rts_mean[0]

	# all_rts_mean.append(rts_mean)



	# plt.show()
	

	# embed()	

	pa.unload_data()

# run_analysis('s2'

# [run_analysis(sub) for sub in sublist]
plt.savefig(os.path.join(figfolder,'all-ev_pupil-amp.pdf'))


# plt.figure()

# plt.subplot(1,2,1)
# plt.title('Pupil amplitude')

# plt.bar(np.arange(4), np.mean(psignals_mean,axis=0), yerr = np.std(psignals_mean,axis=0)/np.sqrt(len(psignals_mean)), color='w')

# plt.axis('tight')
# plt.xticks(np.arange(4)+0.25, ('P','P-U','U-P','U-U'))

# plt.subplot(1,2,2)
# plt.title('Reaction time')

# plt.bar(np.arange(4), np.mean(all_rts_mean,axis=0), yerr = np.std(all_rts_mean,axis=0)/np.sqrt(len(all_rts_mean)), color='w')
# plt.axis('tight')
# plt.xticks(np.arange(4)+0.25, ('P','P-U','U-P','U-U'))

# sn.despine(offset=1)

# plt.savefig(os.path.join(figfolder, 'all-ev_amp-rt.pdf'))
# plt.figure()

# plt.bar([1,2,3],np.mean(all_sub_rts, axis=1))
# plt.plot([1,2,3],all_sub_rts,'o')

# plt.savefig(os.path.join(figfolder,'all_sub_rts.png'))