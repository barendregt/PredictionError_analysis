

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
# from numpy import *
# import scipy as sp
# from pandas import *

from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas.rpy.common as com

from math import *
import os,glob,sys,platform

import pickle as pickle
import pandas as pd

from IPython import embed

from BehaviorAnalyzer import BehaviorAnalyzer
from PupilAnalyzer import PupilAnalyzer
from Plotter import Plotter


from analysis_parameters import *

pl = Plotter(figure_folder = figfolder, linestylemap = linestylemap)



pupil_signals = {'correct': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)},
				 'incorrect': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)}}		 

pupil_signals_pile = {'correct': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)},
				 'incorrect': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)}}		 

reaction_times = {'correct': {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []},
				 'incorrect': {'PP': [],
				 'UP': [],
				 'PU': [],
				 'UU': []}}		 




sublist_pile = {'correct': {'PP': [],
							 'UP': [],
							 'PU': [],
							 'UU': []},
				 'incorrect': {'PP': [],
							 'UP': [],
							 'PU': [],
							 'UU': []}
			 }		 



#		TASK:  		  COLOR	    ORI
condition_keymap = { 0: 'PP',  1: 'PP',
					10: 'PU', 20: 'PU',
					30: 'UP', 40: 'UP',
					50: 'UU', 60: 'UU'}

inverse_keymap = {'PP': [0,1],
				  'UP': [30,40],
				  'PU': [10,20],
				  'UU': [50,60]}

for subname in sublist:

	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	# Initialize PA object
	pa = PupilAnalyzer(subname, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = stimulus_deconvolution_interval, verbosity = 0)


	# Combine signals based on condition	

	sub_signals = {'correct': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)},
								 'incorrect': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
								 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)}}		 

	sub_rts = {'correct': {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []},
					 'incorrect': {'PP': [],
					 'UP': [],
					 'PU': [],
					 'UU': []}}	


	# Get trial-based, event-related, baseline-corrected signals centered on stimulus onset
	pa.signal_per_trial(only_correct = True, only_incorrect = False, reference_phase = 7, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = False, return_rt = True)

	for (key,signals) in pa.trial_signals.items():
		if len(signals)>0:
			sub_signals['correct'][condition_keymap[key]] = np.append(sub_signals['correct'][condition_keymap[key]], signals, axis=0)
			sub_rts['correct'][condition_keymap[key]].extend(pa.trial_rts[key])

	# Get trial-based, event-related, baseline-corrected signals centered on stimulus onset
	pa.signal_per_trial(only_correct = False, only_incorrect = True, reference_phase = 7, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = False, return_rt = True)

	for (key,signals) in pa.trial_signals.items():
		if len(signals)>0:
			sub_signals['incorrect'][condition_keymap[key]] = np.append(sub_signals['incorrect'][condition_keymap[key]], signals, axis=0)
			sub_rts['incorrect'][condition_keymap[key]].extend(pa.trial_rts[key])


	# for con in inverse_keymap.keys():
	# 	pupil_signals['correct'][con] = np.append(pupil_signals['correct'][con], np.mean(sub_signals['correct'][con], axis=0)[np.newaxis,:], axis=0)
	# 	pupil_signals['incorrect'][con] = np.append(pupil_signals['incorrect'][con], np.mean(sub_signals['incorrect'][con], axis=0)[np.newaxis,:], axis=0)

	for con in inverse_keymap.keys():
		pupil_signals['correct'][con] = np.append(pupil_signals['correct'][con], sub_signals['correct'][con].mean(axis=0)[np.newaxis,:], axis=0)
		pupil_signals['incorrect'][con] = np.append(pupil_signals['incorrect'][con], sub_signals['incorrect'][con].mean(axis=0)[np.newaxis,:], axis=0)	

		reaction_times['correct'][con].append(np.median(sub_rts['correct'][con]))
		reaction_times['incorrect'][con].append(np.median(sub_rts['incorrect'][con]))

		# pupil_signals_pile['correct'][con] = np.append(pupil_signals_pile['correct'][con], sub_signals['correct'][con], axis=0)
		# pupil_signals_pile['incorrect'][con] = np.append(pupil_signals_pile['incorrect'][con], sub_signals['incorrect'][con], axis=0)

		# sublist_pile['correct'][con].append([subname]*sub_signals['correct'][con].shape[0])
		# sublist_pile['incorrect'][con].append([subname]*sub_signals['incorrect'][con].shape[0])


error_minus_noerror_correct = {'UP':[],
							   'PU':[],
							   'UU':[]}
error_minus_noerror_incorrect = {'UP':[],
							   'PU':[],
							   'UU':[]}
incorrect_minus_correct =  {'UP':[],
							   'PU':[],
							   'UU':[]}

for con in ['PU','UP','UU']:
	error_minus_noerror_correct[con] = pupil_signals['correct'][con]-pupil_signals['correct']['PP']#.mean(axis=0)

for con in ['PU','UP','UU']:
	error_minus_noerror_incorrect[con] = pupil_signals['incorrect'][con]-pupil_signals['incorrect']['PP']#.mean(axis=0)


error_minus_noerror_correct['UP'] = np.vstack([error_minus_noerror_correct['UP'],error_minus_noerror_correct['UU']])
error_minus_noerror_incorrect['UP'] = np.vstack([error_minus_noerror_incorrect['UP'],error_minus_noerror_incorrect['UU']])

# incorrect_minus_correct['UP'] = combined_incorrect-combined_correct

for con in ['PU','UP']:
	incorrect_minus_correct[con] = error_minus_noerror_incorrect[con] - error_minus_noerror_correct[con]


# embed()

smooth_signal = True
smooth_factor = 50

# pl.open_figure(force=1)

# # pl.subplot(3,2,1)

# pl.event_related_pupil_average(data=pupil_signals['correct'], conditions = ['PP','UP','PU'], signal_labels={'PP':'None','UP':'TaskRel','PU':'TaskIrrel'}, x_lim=[500/smooth_factor,5000/smooth_factor],xticks=np.arange(500/smooth_factor,6000/smooth_factor,500/smooth_factor),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.25,.8],compute_mean=True, compute_sd = True, smooth_signal=smooth_signal, smooth_factor=smooth_factor, show_legend=True,title='Correct')

# pl.save_figure(filename = 'correct_pupil_average.pdf', sub_folder = 'over_subs/pupil')

# # pl.subplot(3,2,2)
# pl.open_figure(force=1)

# pl.event_related_pupil_average(data=pupil_signals['incorrect'], conditions = ['PP','UP','PU'], signal_labels={'PP':'None','UP':'TaskRel','PU':'TaskIrrel'},x_lim=[500/smooth_factor,5000/smooth_factor],xticks=np.arange(500/smooth_factor,6000/smooth_factor,500/smooth_factor),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.25,.8],compute_mean=True, compute_sd = True,smooth_signal=smooth_signal, smooth_factor=smooth_factor, show_legend=True,title='Incorrect')

# pl.save_figure(filename = 'incorrect_pupil_average.pdf', sub_folder = 'over_subs/pupil')

# # pl.subplot(3,2,3)
# pl.open_figure(force=1)

# pl.event_related_pupil_average(data=error_minus_noerror_correct, conditions = ['UP','PU'],signal_labels={'UP':'TaskRel','PU':'TaskIrrel'},x_lim=[500/smooth_factor,5000/smooth_factor],xticks=np.arange(500/smooth_factor,6000/smooth_factor,500/smooth_factor),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.1,0.3],ylabel='Pupil size (difference from None)', compute_mean=True, compute_sd = True,smooth_signal=smooth_signal, smooth_factor=smooth_factor, show_legend=True)

# pl.save_figure(filename = 'correct_pupil_diff.pdf', sub_folder = 'over_subs/pupil')

# # pl.subplot(3,2,4)
# pl.open_figure(force=1)

# pl.event_related_pupil_average(data=error_minus_noerror_incorrect, conditions = ['UP','PU'],signal_labels={'UP':'TaskRel','PU':'TaskIrrel'},x_lim=[500/smooth_factor,5000/smooth_factor],xticks=np.arange(500/smooth_factor,6000/smooth_factor,500/smooth_factor),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.1,0.3],ylabel='Pupil size (difference from None)',compute_mean=True, compute_sd = True,smooth_signal=smooth_signal, smooth_factor=smooth_factor, show_legend=True)

# pl.save_figure(filename = 'incorrect_pupil_diff.pdf', sub_folder = 'over_subs/pupil')

# # pl.subplot(3,1,3)
# pl.open_figure(force=1)

# pl.event_related_pupil_average(data=incorrect_minus_correct, conditions = ['UP','PU'], signal_labels={'UP':'TaskRel','PU':'TaskIrrel'},x_lim=[500/smooth_factor,5000/smooth_factor],xticks=np.arange(500/smooth_factor,6000/smooth_factor,500/smooth_factor),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.1,0.3],compute_mean=True, compute_sd = True, smooth_signal=smooth_signal, smooth_factor=smooth_factor, show_legend=True,title='Difference')

# pl.save_figure(filename = 'incorrect_minus_correct.pdf', sub_folder = 'over_subs/pupil')

#t = np.zeros((5500,1))
#p = np.zeros((5500,1))

# embed()
# pl.open_figure(force=1)

# pl.subplot(1,2,1)

test_condition = 'UP'


combined_rts = np.hstack([reaction_times['correct']['UP'],reaction_times['correct']['UU'],reaction_times['incorrect']['UP'],reaction_times['incorrect']['UU']])

combined_rts_c = np.mean(combined_rts)*np.hstack([reaction_times['correct'][test_condition],reaction_times['correct']['UU']])/np.repeat(reaction_times['correct']['PP'],2)
combined_rts_i = np.mean(combined_rts)*np.hstack([reaction_times['incorrect'][test_condition],reaction_times['incorrect']['UU']])/np.repeat(reaction_times['incorrect']['PP'],2)

# combined_rts_c = np.mean(combined_rts)*np.array(reaction_times['correct'][test_condition])/np.array(reaction_times['correct']['PP'])
# combined_rts_i = np.mean(combined_rts)*np.array(reaction_times['incorrect'][test_condition])/np.array(reaction_times['incorrect']['PP'])

# [rt_low, rt_mean, rt_up] = (signal_sample_frequency/smooth_factor) * np.percentile(combined_rts,[2.5,50,97.5]) + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))

rt_mean = (signal_sample_frequency/smooth_factor) * np.mean(combined_rts) + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))
rt_std = (signal_sample_frequency/smooth_factor) * (1.96*np.std(combined_rts)/sqrt(33))# + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))

rt_low = rt_mean - rt_std
rt_up  = rt_mean + rt_std

rt_mean_c = (signal_sample_frequency/smooth_factor) * np.mean(combined_rts_c) + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))
rt_mean_i = (signal_sample_frequency/smooth_factor) * np.mean(combined_rts_i) + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))

rt_std_c = (signal_sample_frequency/smooth_factor) * (np.std(combined_rts_c)/sqrt(33))# + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))
rt_std_i = (signal_sample_frequency/smooth_factor) * (np.std(combined_rts_i)/sqrt(33))# + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))


pl.open_figure(force=1)

pl.linestylemap[test_condition] = ['k','-',None,None,None]

pl.hline(0.0, linewidth=0.5, color='k',linestyle='dotted')
pl.vline(x=rt_mean_c,linewidth=2,color='g',alpha=0.75,linestyle='solid')
plt.legend(loc='best')
plt.fill_betweenx(np.arange(-0.2,0.4,0.1),rt_mean_c-rt_std_c,rt_mean_c+rt_std_c,color='g',alpha=0.2)

pl.vline(x=rt_mean_i,linewidth=2,color='r',alpha=0.75,linestyle='solid')
plt.fill_betweenx(np.arange(-0.2,0.4,0.1),rt_mean_i-rt_std_i,rt_mean_i+rt_std_i,color='r',alpha=0.2)

pl.event_related_pupil_average(data=incorrect_minus_correct, conditions=[test_condition], signal_labels={'UP':'TaskRel','PU':'TaskIrrel','UU':'both'},x_lim=[500/smooth_factor,5000/smooth_factor],xticks=np.arange(500/smooth_factor,6000/smooth_factor,500/smooth_factor),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.2,0.3],compute_mean=True, compute_sd = True, smooth_signal=smooth_signal, with_stats=True, sig_marker_ypos = -0.05, smooth_factor=smooth_factor, show_legend=True,title='Difference incorrect vs correct')
pl.save_figure(filename = '%s_diff_stats.pdf'%test_condition, sub_folder = 'over_subs/pupil')




test_condition = 'PU'


combined_rts = np.hstack([reaction_times['correct'][test_condition],reaction_times['incorrect'][test_condition]])

combined_rts_c = np.mean(combined_rts)*np.array(reaction_times['correct'][test_condition])/np.array(reaction_times['correct']['PP'])
combined_rts_i = np.mean(combined_rts)*np.array(reaction_times['incorrect'][test_condition])/np.array(reaction_times['incorrect']['PP'])

# [rt_low, rt_mean, rt_up] = (signal_sample_frequency/smooth_factor) * np.percentile(combined_rts,[2.5,50,97.5]) + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))

rt_mean = (signal_sample_frequency/smooth_factor) * np.mean(combined_rts) + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))
rt_std = (signal_sample_frequency/smooth_factor) * (1.96*np.std(combined_rts)/sqrt(33))# + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))

rt_low = rt_mean - rt_std
rt_up  = rt_mean + rt_std

rt_mean_c = (signal_sample_frequency/smooth_factor) * np.mean(combined_rts_c) + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))
rt_mean_i = (signal_sample_frequency/smooth_factor) * np.mean(combined_rts_i) + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))

rt_std_c = (signal_sample_frequency/smooth_factor) * (np.std(combined_rts_c)/sqrt(33))# + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))
rt_std_i = (signal_sample_frequency/smooth_factor) * (np.std(combined_rts_i)/sqrt(33))# + ((signal_sample_frequency/smooth_factor)*abs(stimulus_deconvolution_interval[0]))


pl.open_figure(force=1)

pl.linestylemap[test_condition] = ['k','-',None,None,None]
pl.hline(0.0, linewidth=0.5, color='k',linestyle='dotted')
pl.vline(x=rt_mean_c,linewidth=2,color='g',alpha=0.75,linestyle='solid')
plt.legend(loc='best')
plt.fill_betweenx(np.arange(-0.2,0.4,0.1),rt_mean_c-rt_std_c,rt_mean_c+rt_std_c,color='g',alpha=0.2)

pl.vline(x=rt_mean_i,linewidth=2,color='r',alpha=0.75,linestyle='solid')
plt.fill_betweenx(np.arange(-0.2,0.4,0.1),rt_mean_i-rt_std_i,rt_mean_i+rt_std_i,color='r',alpha=0.2)

pl.event_related_pupil_average(data=incorrect_minus_correct, conditions=[test_condition], signal_labels={'UP':'TaskRel','PU':'TaskIrrel','UU':'both'},x_lim=[500/smooth_factor,5000/smooth_factor],xticks=np.arange(500/smooth_factor,6000/smooth_factor,500/smooth_factor),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.2,0.3],compute_mean=True, compute_sd = True, smooth_signal=smooth_signal, with_stats=True, sig_marker_ypos = -0.05, smooth_factor=smooth_factor, show_legend=True,title='Difference incorrect vs correct')
pl.save_figure(filename = '%s_diff_stats.pdf'%test_condition, sub_folder = 'over_subs/pupil')