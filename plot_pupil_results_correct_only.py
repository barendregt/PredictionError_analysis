import numpy as np
import scipy as sp
import seaborn as sn
sn.set(style='ticks')
import matplotlib.pyplot as pl
# %matplotlib inline
from math import *
import os,glob,datetime

import cPickle as pickle
import pandas as pd

from BehaviorAnalyzer import BehaviorAnalyzer
from IPython import embed



raw_data_folder = '/home/barendregt/Projects/Attention_Prediction/Psychophysics/Data' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/Attention_Prediction/Figures'

sublist = ['s3']#['s1','s2','s3','s4','s5','s6']#,'s2','s3','s4','s5']#['s1','s2','s4']
down_fs = 100
winlength = 5000#6500
minwinlength = 4500#6000

def sigmoid(x, a, b, l):
	g = 0.5
	# l = 0.1
	# l=0
	return g + (1-g-l)/(1+np.exp(-(x-a)/b))

 
for subii,subname in enumerate(sublist):
	

	# embed()
	rawfolder = os.path.join(raw_data_folder,subname)
	
	# tmp = pickle.load(open(os.path.join(rawfolder,'pupil_' + subname + '_databin.p'),'rb'))

	# task_data = [0]
	# events = tmp[1]
	# # task_performance = tmp[2]
	# trial_signals = tmp[2]
	# fir_signals = tmp[3]

	task_data, events, trial_signals, fir_signals, pupil_signal_trials = pickle.load(open(os.path.join(rawfolder,'pupil_' + subname + '_databin.p'),'rb'))
	task_performance = pickle.load(open(os.path.join(rawfolder,'behavior_' + subname + '_databin.p'),'rb'))[2]


	embed()

	correct_answer = np.array(task_data['trial_correct'])
	trial_codes = np.array(task_data['coded_trials'])

	# pl.show()

	f = pl.figure()

	s1 = f.add_subplot(221)	

	s1.set_title('Pupil time course')

	labels = ['Pp','Pu','Up','Uu']#['Expected','Unexpected']
	colors = ['k','r','g','b']#[.6, .6, .6]]
	lstyles = ['solid','solid']

	pred_signals = []
	unpred_signals = []

	psignals = [[],[],[],[]]#p_signals = pup_signals = upp_signals = upup_signals = []


	for name in events['codes']:

		# iis = np.array(correct_answer[trial_codes==name]==1)

		signal = np.mean(trial_signals[name]['stim'],axis=0)

		if name < 10:
			#psignals[0].extend(signal)
			pred_signals.append(signal)
		# elif name <30:
		# 	psignals[1].extend(signal)
		# elif name <50:
		# 	psignals[2].extend(signal)
		else:
			# psignals[3].extend(signal)
			unpred_signals.append(signal)



	#pl.axvline(x=[0,1.25,2.5])
	# pl.axvline(x=5,color='k',linewidth=0.75)
	# pl.axvline(x=12.5,color='k',linewidth=0.75)
	# pl.axvline(x=25,color='k',linewidth=0.5)
	# pl.axvline(x=30,color='k',linewidth=1)
	# pl.axvline(x=33.3,color='k',linewidth=0.75)

	# for pii in range(len(psignals)):
	# 	pl.plot(np.nanmean(psignals[pii],0),
	# 		    color = colors[pii],
	# 		    label = labels[pii])

	# pl.fill_between(np.linspace(-.5,(minwinlength/1000)-.5,np.size(np.mean(pred_signals,axis=0))),
	# 				1.96*np.mean(pred_signals,0)-(np.std(pred_signals,0)/sqrt(np.size(pred_signals))), 
	# 				1.96*np.mean(pred_signals,0)+(np.std(pred_signals,0)/sqrt(np.size(pred_signals))),
	# 				color = 'k',
	# 				alpha = 0.1)

	pl.plot(np.nanmean(pred_signals,0),
		    color = colors[1],
		    label = 'Unexpected')	
	pl.plot(np.nanmean(unpred_signals,0),
		    color = colors[1],
		    label = 'Unexpected')	





	# pl.fill_between(np.linspace(-.5,(minwinlength/1000)-.5,np.size(np.mean(unpred_signals,axis=0))),
	# 				1.96*np.mean(unpred_signals,0)-(np.std(unpred_signals,0)/sqrt(np.size(unpred_signals))), 
	# 				1.96*np.mean(unpred_signals,0)+(np.std(unpred_signals,0)/sqrt(np.size(unpred_signals))),
	# 				color = 'k',
	# 				alpha = 0.1)

	pl.legend()

	pl.xlabel('Time after stim onset (s)')
	pl.ylabel('Pupil size')

	sn.despine(offset=5)	

	# embed()

	s = f.add_subplot(222)

	s.set_title('Pupil response')

	labels = ['Expected','Unexpected']
	colors = ['k',[.6, .6, .6]]
	lstyles = ['solid','solid']

	pred_peaks = []
	unpred_peaks = []

	peak_window = [6,14]

	for name in trial_signals.keys():

		msignal = np.array(trial_signals[name]['stim']).mean(axis=0)

		peak_signals = np.array([(signal*msignal)/(np.linalg.norm(msignal, ord=2)**2) for signal in trial_signals[name]['stim']])[:,peak_window[0]:peak_window[1]].sum(axis=1)
		if name < 10:
			pred_peaks.extend(peak_signals)
		else:
			# peak_signals = np.array([(signal*msignal)/(np.linalg.norm(msignal, ord=2)**2) for signal in trial_signals[name]['stim']])[:,peak_window[0]:peak_window[1]].max(axis=1)
			unpred_peaks.extend(peak_signals)

	pred_peaks = np.array(pred_peaks)
	unpred_peaks = np.array(unpred_peaks)

	width = 0.5

	pl.bar(0.75, 
		   np.mean(pred_peaks), 
		   edgecolor = colors[0],
		   facecolor = 'w',
		   label = 'Expected',
		   width = width,
		   linewidth = 2,
		   yerr = 1.96*np.std(pred_peaks)/np.sqrt(pred_peaks.size))

	# for p in pred_peaks:
	# 	pl.plot(1,p,'o')

	pl.bar(1.75, 
		   np.mean(unpred_peaks), 
		   edgecolor = colors[1],
		   facecolor = 'w',
		   label = 'Unexpected',
		   width = width,
		   linewidth = 2,
		   yerr = 1.96*np.std(unpred_peaks)/np.sqrt(unpred_peaks.size))	


	# for p in unpred_peaks:
	# 	pl.plot(2,p,'o')

	pl.xticks([1,2],labels)
	# pl.errorbar(labels, np.mean([pred_peaks, unpred_peaks], axis=0), np.std([pred_peaks, unpred_peaks], axis=0))


	s = f.add_subplot(223)

	s.set_title('Reaction time')

	# task_performance['pred-v-unpred-rt'][:,1]


	

	s = f.add_subplot(224)

	s.set_title('Performance')


	for ii in range(len(task_performance['pred-v-unpred'])):

		xs = np.linspace(min(task_performance['pred-v-unpred'][0][:,0]),
					     max(task_performance['pred-v-unpred'][0][:,0]),1000)	

		pl.semilogx(task_performance['pred-v-unpred'][ii][:,0],
				task_performance['pred-v-unpred'][ii][:,1],
				'o',
				color=sn.desaturate(colors[ii],1),
				alpha = 0.25)
		
		pl.semilogx(xs, 
			    sigmoid(xs, task_performance['psych_curve_params']['pred-v-unpred'][ii][0], task_performance['psych_curve_params']['pred-v-unpred'][ii][1], task_performance['psych_curve_params']['pred-v-unpred'][ii][2]), 
			    color=sn.desaturate(colors[ii],1), 
			    linestyle=lstyles[ii],
			    alpha=1,
			    label=labels[ii])


	# pl.xlabel('Intensity difference (stim2-stim1)')
	pl.ylabel('Performance (% correct)')

	pl.ylim([.5, 1])
	pl.xlim([min(task_performance['pred-v-unpred'][0][:,0]), max(task_performance['pred-v-unpred'][0][:,0])])

	pl.xticks(task_performance['pred-v-unpred'][0][:,0], [0.25,0.35,0.5, 0.7, 1.0, 1.41, 2.0, 2.83, 4])

	pl.legend(loc = 'upper left')
	sn.despine(offset=5)


	pl.tight_layout()

	f.savefig(os.path.join(figfolder,'all', subname + '_beh_pupil' + '.pdf'))# + '-' + str(datetime.datetime.now()) + '.pdf'))

	# f.savefig(os.path.join(figfolder, subname + '_fir_pupil' + '.pdf'))# + '-' + str(datetime.datetime.now()) + '.pdf'))
	# # f.close()
	pl.close()