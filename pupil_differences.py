

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
				  'UP': [30,40,50,60],
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

		reaction_times['correct'][con].append(np.mean(sub_rts['correct'][con]))
		reaction_times['incorrect'][con].append(np.mean(sub_rts['incorrect'][con]))

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

rt_mean = np.mean(np.hstack([reaction_times['correct']['UP'],reaction_times['correct']['UU']]))
rt_std = np.std(np.hstack([reaction_times['correct']['UP'],reaction_times['correct']['UU']]))/np.sqrt(66)

rt_low = rt_mean - rt_std
rt_up  = rt_mean + rt_std

pl.open_figure(force=1)
pl.event_related_pupil_average(data=incorrect_minus_correct, conditions=['UP'], signal_labels={'UP':'TaskRel','PU':'TaskIrrel','UU':'both'},x_lim=[500/smooth_factor,5000/smooth_factor],xticks=np.arange(500/smooth_factor,6000/smooth_factor,500/smooth_factor),xticklabels=np.arange(-0.5,4.5,0.5),y_lim=[-0.1,0.3],compute_mean=True, compute_sd = True, smooth_signal=smooth_signal, with_stats=True, sig_marker_ypos = -0.05, smooth_factor=smooth_factor, show_legend=True,title='Difference')
pl.vline(x=rt_mean,linewidth=1,color='k',alpha=0.75)
pl.vline(x=rt_low,linewidth=1,color='k')
pl.vline(x=rt_up,linewidth=1,color='k')
pl.save_figure(filename = 'UP_diff_stats.pdf', sub_folder = 'over_subs/pupil')

# plt.show()
# embed()

# ds_pupil_signals = {'correct': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*(signal_sample_frequency/smooth_factor))),dtype=float),
# 				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*(signal_sample_frequency/smooth_factor))),dtype=float),
# 				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*(signal_sample_frequency/smooth_factor))),dtype=float),
# 				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimuflus_deconvolution_interval[0])*(signal_sample_frequency/smooth_factor))),dtype=float)},
# 				 'incorrect': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*(signal_sample_frequency/smooth_factor))),dtype=float),
# 				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*(signal_sample_frequency/smooth_factor))),dtype=float),
# 				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*(signal_sample_frequency/smooth_factor))),dtype=float),
# 				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*(signal_sample_frequency/smooth_factor))),dtype=float)}}	

# ds_pupil_signals = {'correct': [], 'incorrect': []}

# # downsample data to reduce computation time
# for con in ['PP','PU','UP','UU']:
# 	ds_pupil_signals['correct'].append(sp.signal.decimate(pupil_signals['correct'][con], smooth_factor, 1))
# 	ds_pupil_signals['incorrect'].append(sp.signal.decimate(pupil_signals['incorrect'][con], smooth_factor, 1))

# ds_pupil_signals['correct'] = np.dstack(ds_pupil_signals['correct'])
# ds_pupil_signals['incorrect'] = np.dstack(ds_pupil_signals['incorrect'])

# # Export data to R
# for t in range(ds_pupil_signals['correct'].shape[1]):
# 	tdata = pd.DataFrame({'pupil': np.hstack([ds_pupil_signals['correct'][:,t,:].ravel(),ds_pupil_signals['incorrect'][:,t,:].ravel()]), 'TR': [0,0,1,1]*len(sublist)*2, 'TI': [0,1,0,1]*len(sublist)*2, 'PredError': [0, 1, 1, 1]*len(sublist)*2, 'TaskRel': [0, 0, 1, 1]*len(sublist)*2, 'Sub': np.hstack(np.tile(np.repeat(sublist, 4),(2,1))), 'type': np.repeat(['correct','incorrect'],4*len(sublist))})

# 	tdata.to_csv('data_output/pupil_data_avg_t%i.csv'%t)



# ds_pupil_signals = {'correct': [], 'incorrect': []}

# # downsample data to reduce computation time
# # for con in ['PP','PU','UP']:
# ds_pupil_signals['correct'] = sp.signal.decimate(pupil_signals_pile['correct']['UP'], smooth_factor, 1)
# ds_pupil_signals['incorrect'] = sp.signal.decimate(pupil_signals_pile['incorrect']['UP'], smooth_factor, 1)

# # ds_pupil_signals['correct'] = np.dstack(ds_pupil_signals['correct'])
# # ds_pupil_signals['incorrect'] = np.dstack(ds_pupil_signals['incorrect'])

# # Export data to R
# for t in range(ds_pupil_signals['correct'].shape[1]):
# 	tdata = pd.DataFrame({'pupil': np.hstack([ds_pupil_signals['correct'][:,t].ravel(),ds_pupil_signals['incorrect'][:,t].ravel()]), 'Sub': np.hstack([np.hstack(sublist_pile['correct']['UP']),np.hstack(sublist_pile['incorrect']['UP'])]), 'type': np.repeat(['correct','incorrect'],[np.hstack(sublist_pile['correct']['UP']).shape[0],np.hstack(sublist_pile['incorrect']['UP']).shape[0]])})

# 	tdata.to_csv('data_output/pupil_data_UP_single_t%i.csv'%t)
