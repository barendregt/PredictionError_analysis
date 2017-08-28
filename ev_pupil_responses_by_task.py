

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

import pickle as pickle
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

response_pupil_signals = {'col_PP': [],
						 'col_UP': [],
						 'col_PU': [],
						 'col_UU': [],
						 'ori_PP': [],
						 'ori_UP': [],
						 'ori_PU': [],
						 'ori_UU': []}
stimulus_pupil_signals = {'col_PP': [],
						 'col_UP': [],
						 'col_PU': [],
						 'col_UU': [],
						 'ori_PP': [],
						 'ori_UP': [],
						 'ori_PU': [],
						 'ori_UU': []}				 
response_diff_signals  = {'col_PP': [],
						 'col_UP': [],
						 'col_PU': [],
						 'col_UU': [],
						 'ori_PP': [],
						 'ori_UP': [],
						 'ori_PU': [],
						 'ori_UU': []}	
stimulus_diff_signals  = {'col_PP': [],
						 'col_UP': [],
						 'col_PU': [],
						 'col_UU': [],
						 'ori_PP': [],
						 'ori_UP': [],
						 'ori_PU': [],
						 'ori_UU': []}				 


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
	col_ref_signals = []
	ori_ref_signals = []

	for key,trial_signal in list(pa.trial_signals.items()):
		if key < 10:
			#trial_signal = trial_signal - trial_signal[:,zero_point][:,np.newaxis]

			ref_signals.extend(trial_signal)

			if key == 0:
				col_ref_signals.extend(trial_signal)
			else:
				ori_ref_signals.extend(trial_signal)

	msignal = np.mean(ref_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	ori_msignal = np.mean(ori_ref_signals, axis=0)
	col_msignal = np.mean(col_ref_signals, axis=0)

	pp_signal = []
	up_signal = []
	pu_signal = []
	uu_signal = []
	

	try:
		for key, trial_signal in list(pa.trial_signals.items()):
			if len(trial_signal)>0:
				if key == 0:
					response_pupil_signals['col_PP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
				elif key == 1:
					response_pupil_signals['ori_PP'].extend(trial_signal)
				elif key == 10:
					response_pupil_signals['col_PU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
				elif key == 20:
					response_pupil_signals['ori_PU'].extend(trial_signal)				
				elif key == 30:
					response_pupil_signals['col_UP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
				elif key == 40:
					response_pupil_signals['ori_UP'].extend(trial_signal)	
				elif key == 50:
					response_pupil_signals['col_UU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
				elif key == 60:
					response_pupil_signals['ori_UU'].extend(trial_signal)	


			if len(trial_signal)>0:

				if key == 0:
					response_diff_signals['col_PP'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
				elif key == 1:
					response_diff_signals['ori_PP'].extend(trial_signal - ori_msignal)
				elif key == 10:
					response_diff_signals['col_PU'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
				elif key == 20:
					response_diff_signals['ori_PU'].extend(trial_signal - ori_msignal)				
				elif key == 30:
					response_diff_signals['col_UP'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
				elif key == 40:
					response_diff_signals['ori_UP'].extend(trial_signal - ori_msignal)	
				elif key == 50:
					response_diff_signals['col_UU'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
				elif key == 60:
					response_diff_signals['ori_UU'].extend(trial_signal - ori_msignal)
	except:
		embed()


	pa.deconvolution_interval = stimulus_deconvolution_interval
	pa.signal_per_trial(only_correct = True, only_incorrect = False, reference_phase = 4, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = True)

	# pa.get_IRF()

	# all_sub_IRF['stimulus'].append(pa.sub_IRF['stimulus'])

	# embed()
	ref_signals = []
	col_ref_signals = []
	ori_ref_signals = []

	for key,trial_signal in list(pa.trial_signals.items()):
		if key < 10:
			#trial_signal = trial_signal - trial_signals[:,zero_point][:,np.newaxis]

			ref_signals.extend(trial_signal)

			if key == 0:
				col_ref_signals.extend(trial_signal)
			else:
				ori_ref_signals.extend(trial_signal)

	msignal = np.mean(ref_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	ori_msignal = np.mean(ori_ref_signals, axis=0)
	col_msignal = np.mean(col_ref_signals, axis=0)

	pp_signal = []
	up_signal = []
	pu_signal = []
	uu_signal = []
	

	for key, trial_signal in list(pa.trial_signals.items()):
		if len(trial_signal)>0:
			if key == 0:
				stimulus_pupil_signals['col_PP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
			elif key == 1:
				stimulus_pupil_signals['ori_PP'].extend(trial_signal)
			elif key == 10:
				stimulus_pupil_signals['col_PU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
			elif key == 20:
				stimulus_pupil_signals['ori_PU'].extend(trial_signal)				
			elif key == 30:
				stimulus_pupil_signals['col_UP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
			elif key == 40:
				stimulus_pupil_signals['ori_UP'].extend(trial_signal)	
			elif key == 50:
				stimulus_pupil_signals['col_UU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
			elif key == 60:
				stimulus_pupil_signals['ori_UU'].extend(trial_signal)	


		if len(trial_signal)>0:

			if key == 0:
				stimulus_diff_signals['col_PP'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
			elif key == 1:
				stimulus_diff_signals['ori_PP'].extend(trial_signal - ori_msignal)
			elif key == 10:
				stimulus_diff_signals['col_PU'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
			elif key == 20:
				stimulus_diff_signals['ori_PU'].extend(trial_signal - ori_msignal)				
			elif key == 30:
				stimulus_diff_signals['col_UP'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
			elif key == 40:
				stimulus_diff_signals['ori_UP'].extend(trial_signal - ori_msignal)	
			elif key == 50:
				stimulus_diff_signals['col_UU'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
			elif key == 60:
				stimulus_diff_signals['ori_UU'].extend(trial_signal - ori_msignal)


# embed()
pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_pupil_signals, conditions = ['col_PP','col_UP','col_PU','col_UU'], signal_labels = {'col_PP': 'Predicted', 'col_UP': 'Task relevant','col_PU':'Task irrelevant','col_UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True)

pl.save_figure('col_pupil_response_button-press.pdf', sub_folder = 'over_subs/pupil')


pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_diff_signals, conditions = ['col_UP','col_PU','col_UU'], signal_labels = {'col_PP': 'Predicted', 'col_UP': 'Task relevant','col_PU':'Task irrelevant','col_UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True)

pl.save_figure('col_pupil_difference_button-press.pdf', sub_folder = 'over_subs/pupil')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_pupil_signals, conditions = ['col_PP','col_UP','col_PU','col_UU'], signal_labels = {'col_PP': 'Predicted', 'col_UP': 'Task relevant','col_PU':'Task irrelevant','col_UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True)

pl.save_figure('col_pupil_response-stimulus.pdf', sub_folder = 'over_subs/pupil')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_diff_signals, conditions = ['col_UP','col_PU','col_UU'], signal_labels = {'col_PP': 'Predicted', 'col_UP': 'Task relevant','col_PU':'Task irrelevant','col_UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True)

pl.save_figure('col_pupil_difference-stimulus.pdf', sub_folder = 'over_subs/pupil')


pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_pupil_signals, conditions = ['ori_PP','ori_UP','ori_PU','ori_UU'], signal_labels = {'ori_PP': 'Predicted', 'ori_UP': 'Task relevant','ori_PU':'Task irrelevant','ori_UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True)

pl.save_figure('ori_pupil_response_button-press.pdf', sub_folder = 'over_subs/pupil')


pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_diff_signals, conditions = ['ori_UP','ori_PU','ori_UU'], signal_labels = {'ori_PP': 'Predicted', 'ori_UP': 'Task relevant','ori_PU':'Task irrelevant','ori_UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True)

pl.save_figure('ori_pupil_difference_button-press.pdf', sub_folder = 'over_subs/pupil')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_pupil_signals, conditions = ['ori_PP','ori_UP','ori_PU','ori_UU'], signal_labels = {'ori_PP': 'Predicted', 'ori_UP': 'Task relevant','ori_PU':'Task irrelevant','ori_UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True)

pl.save_figure('ori_pupil_response-stimulus.pdf', sub_folder = 'over_subs/pupil')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_diff_signals, conditions = ['ori_UP','ori_PU','ori_UU'], signal_labels = {'ori_PP': 'Predicted', 'ori_UP': 'Task relevant','ori_PU':'Task irrelevant','ori_UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True)

pl.save_figure('ori_pupil_difference-stimulus.pdf', sub_folder = 'over_subs/pupil')






#### PLOT AVERAGES OVER SUBS

# pl.open_figure()

# pl.subplot(1,2,2, title= 'Average pupil difference')

response_pupil_signals = {'col_PP': [],
						 'col_UP': [],
						 'col_PU': [],
						 'col_UU': [],
						 'ori_PP': [],
						 'ori_UP': [],
						 'ori_PU': [],
						 'ori_UU': []}
stimulus_pupil_signals = {'col_PP': [],
						 'col_UP': [],
						 'col_PU': [],
						 'col_UU': [],
						 'ori_PP': [],
						 'ori_UP': [],
						 'ori_PU': [],
						 'ori_UU': []}				 
response_diff_signals  = {'col_PP': [],
						 'col_UP': [],
						 'col_PU': [],
						 'col_UU': [],
						 'ori_PP': [],
						 'ori_UP': [],
						 'ori_PU': [],
						 'ori_UU': []}	
stimulus_diff_signals  = {'col_PP': [],
						 'col_UP': [],
						 'col_PU': [],
						 'col_UU': [],
						 'ori_PP': [],
						 'ori_UP': [],
						 'ori_PU': [],
						 'ori_UU': []}				 


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
	col_ref_signals = []
	ori_ref_signals = []

	for key,trial_signal in list(pa.trial_signals.items()):
		if key < 10:
			#trial_signal = trial_signal - trial_signal[:,zero_point][:,np.newaxis]

			ref_signals.extend(trial_signal)

			if key == 0:
				col_ref_signals.extend(trial_signal)
			else:
				ori_ref_signals.extend(trial_signal)

	msignal = np.mean(ref_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	ori_msignal = np.mean(ori_ref_signals, axis=0)
	col_msignal = np.mean(col_ref_signals, axis=0)

	pp_signal = []
	up_signal = []
	pu_signal = []
	uu_signal = []
	
    
	try:
		for key, trial_signal in list(pa.trial_signals.items()):
			if len(trial_signal)>0:
				if key == 0:
					response_pupil_signals['col_PP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
				elif key == 1:
					response_pupil_signals['ori_PP'].extend(trial_signal)
				elif key == 10:
					response_pupil_signals['col_PU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
				elif key == 20:
					response_pupil_signals['ori_PU'].extend(trial_signal)				
				elif key == 30:
					response_pupil_signals['col_UP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
				elif key == 40:
					response_pupil_signals['ori_UP'].extend(trial_signal)	
				elif key == 50:
					response_pupil_signals['col_UU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
				elif key == 60:
					response_pupil_signals['ori_UU'].extend(trial_signal)	


			if len(trial_signal)>0:

				if key == 0:
					response_diff_signals['col_PP'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
				elif key == 1:
					response_diff_signals['ori_PP'].extend(trial_signal - ori_msignal)
				elif key == 10:
					response_diff_signals['col_PU'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
				elif key == 20:
					response_diff_signals['ori_PU'].extend(trial_signal - ori_msignal)				
				elif key == 30:
					response_diff_signals['col_UP'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
				elif key == 40:
					response_diff_signals['ori_UP'].extend(trial_signal - ori_msignal)	
				elif key == 50:
					response_diff_signals['col_UU'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
				elif key == 60:
					response_diff_signals['ori_UU'].extend(trial_signal - ori_msignal)
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
	col_ref_signals = []
	ori_ref_signals = []

	for key,trial_signal in list(pa.trial_signals.items()):
		if key < 10:
			#trial_signal = trial_signal - trial_signals[:,zero_point][:,np.newaxis]

			ref_signals.extend(trial_signal)

			if key == 0:
				col_ref_signals.extend(trial_signal)
			else:
				ori_ref_signals.extend(trial_signal)

	msignal = np.mean(ref_signals, axis=0)
	msignal_norm = np.linalg.norm(msignal, ord=2)**2

	ori_msignal = np.mean(ori_ref_signals, axis=0)
	col_msignal = np.mean(col_ref_signals, axis=0)

	pp_signal = []
	up_signal = []
	pu_signal = []
	uu_signal = []
	

	for key, trial_signal in list(pa.trial_signals.items()):
		if len(trial_signal)>0:
			if key == 0:
				stimulus_pupil_signals['col_PP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
			elif key == 1:
				stimulus_pupil_signals['ori_PP'].extend(trial_signal)
			elif key == 10:
				stimulus_pupil_signals['col_PU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
			elif key == 20:
				stimulus_pupil_signals['ori_PU'].extend(trial_signal)				
			elif key == 30:
				stimulus_pupil_signals['col_UP'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
			elif key == 40:
				stimulus_pupil_signals['ori_UP'].extend(trial_signal)	
			elif key == 50:
				stimulus_pupil_signals['col_UU'].extend(trial_signal)#.append(np.mean(trial_signal, axis=0))
			elif key == 60:
				stimulus_pupil_signals['ori_UU'].extend(trial_signal)	


		if len(trial_signal)>0:

			if key == 0:
				stimulus_diff_signals['col_PP'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
			elif key == 1:
				stimulus_diff_signals['ori_PP'].extend(trial_signal - ori_msignal)
			elif key == 10:
				stimulus_diff_signals['col_PU'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
			elif key == 20:
				stimulus_diff_signals['ori_PU'].extend(trial_signal - ori_msignal)				
			elif key == 30:
				stimulus_diff_signals['col_UP'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
			elif key == 40:
				stimulus_diff_signals['ori_UP'].extend(trial_signal - ori_msignal)	
			elif key == 50:
				stimulus_diff_signals['col_UU'].extend(trial_signal - col_msignal)#.append(np.mean(trial_signal, axis=0))
			elif key == 60:
				stimulus_diff_signals['ori_UU'].extend(trial_signal - ori_msignal)

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_pupil_signals, conditions = ['col_PP','col_UP','col_PU','col_UU'], signal_labels = {'col_PP': 'Predicted', 'col_UP': 'Task relevant','col_PU':'Task irrelevant','col_UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True)

pl.save_figure('inc_col_pupil_response_button-press.pdf', sub_folder = 'over_subs/pupil/incorrect')


pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_diff_signals, conditions = ['col_UP','col_PU','col_UU'], signal_labels = {'col_PP': 'Predicted', 'col_UP': 'Task relevant','col_PU':'Task irrelevant','col_UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True)

pl.save_figure('inc_col_pupil_difference_button-press.pdf', sub_folder = 'over_subs/pupil/incorrect')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_pupil_signals, conditions = ['col_PP','col_UP','col_PU','col_UU'], signal_labels = {'col_PP': 'Predicted', 'col_UP': 'Task relevant','col_PU':'Task irrelevant','col_UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True)

pl.save_figure('inc_col_pupil_response-stimulus.pdf', sub_folder = 'over_subs/pupil/incorrect')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_diff_signals, conditions = ['col_UP','col_PU','col_UU'], signal_labels = {'col_PP': 'Predicted', 'col_UP': 'Task relevant','col_PU':'Task irrelevant','col_UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True)

pl.save_figure('inc_col_pupil_difference-stimulus.pdf', sub_folder = 'over_subs/pupil/incorrect')


pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_pupil_signals, conditions = ['ori_PP','ori_UP','ori_PU','ori_UU'], signal_labels = {'ori_PP': 'Predicted', 'ori_UP': 'Task relevant','ori_PU':'Task irrelevant','ori_UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(response_deconvolution_interval[0], response_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True)

pl.save_figure('inc_ori_pupil_response_button-press.pdf', sub_folder = 'over_subs/pupil/incorrect')


pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = response_diff_signals, conditions = ['ori_UP','ori_PU','ori_UU'], signal_labels = {'ori_PP': 'Predicted', 'ori_UP': 'Task relevant','ori_PU':'Task irrelevant','ori_UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True)

pl.save_figure('inc_ori_pupil_difference_button-press.pdf', sub_folder = 'over_subs/pupil/incorrect')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_pupil_signals, conditions = ['ori_PP','ori_UP','ori_PU','ori_UU'], signal_labels = {'ori_PP': 'Predicted', 'ori_UP': 'Task relevant','ori_PU':'Task irrelevant','ori_UU':'Both'}, show_legend=True, ylabel = 'Pupil size', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.3, .8], compute_mean = True, compute_sd = True)

pl.save_figure('inc_ori_pupil_response-stimulus.pdf', sub_folder = 'over_subs/pupil/incorrect')

pl.open_figure(force=1)
pl.hline(y=0)
pl.event_related_pupil_average(data = stimulus_diff_signals, conditions = ['ori_UP','ori_PU','ori_UU'], signal_labels = {'ori_PP': 'Predicted', 'ori_UP': 'Task relevant','ori_PU':'Task irrelevant','ori_UU':'Both'}, show_legend=True, ylabel = 'Pupil size difference from noPE', x_lim = [0.5*(signal_sample_frequency/down_fs), 4.5*(signal_sample_frequency/down_fs)], xticks = np.arange(0,5*(signal_sample_frequency/down_fs),0.5*(signal_sample_frequency/down_fs)), xticklabels = np.arange(stimulus_deconvolution_interval[0], stimulus_deconvolution_interval[1],.5), y_lim = [-.10, .25], compute_mean = True, compute_sd = True)

pl.save_figure('inc_ori_pupil_difference-stimulus.pdf', sub_folder = 'over_subs/pupil/incorrect')