from __future__ import division

import numpy as np
import scipy as sp

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from GeneralLinearModel import GeneralLinearModel as GLM

from numpy import *
import scipy as sp
from pandas import *
# import readline
# from rpy2.robjects.packages import importr
# import rpy2.robjects as ro
# import pandas.rpy.common as com
# stats = importr('stats')
# base = importr('base')

import matplotlib.pyplot as plt 
import seaborn as sn 


sn.set(style='ticks', font='Arial', font_scale=1, rc={
	'axes.linewidth': 0.50, 
	'axes.labelsize': 7, 
	'axes.titlesize': 7, 
	'xtick.labelsize': 6, 
	'ytick.labelsize': 6, 
	'legend.fontsize': 6, 
	'xtick.major.width': 0.25, 
	'ytick.major.width': 0.25,
	'text.color': 'Black',
	'axes.labelcolor':'Black',
	'xtick.color':'Black',
	'ytick.color':'Black',} )

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

sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV']#,'BA','BB','BC','BD','BE','BF','BG','BH']#
# sublist = ['AA','AB','AC','AD','AF','AG','AH','AI','AJ','AM']
# sublist_pos = ['AA','AB','AG','AJ','AL','AM','AO','AC','AF','AH','AI','AK','AN','AO','AP']
# sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AD','AE','AK','AL','AM','AN']
sbsetting = [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True]

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
trial_deconvolution_interval = np.array([-3, 5])
# trial_deconvolution_interval = np.array([-1, 3])

down_fs = 100

num_of_events = 3

# linestylemap = {'PP': ['k-'],
# 				 'UP': [''],
# 				 'PU': [''],
# 				 'UU': ['']}

pl = Plotter(figure_folder = figfolder)



all_betas = {'PP': [],
						 'UP': [],
						 'PU': [],
						 'UU': []}

for subname in sublist:

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	#pl.figure_folder = os.path.join(rawfolder,'results/')

	# if not os.path.isdir(os.path.join(rawfolder,'results/')):
	# 	os.makedirs(os.path.join(rawfolder,'results/'))

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = trial_deconvolution_interval, verbosity = 0)

	#pa.recombine_signal_blocks(force_rebuild=True)

	recorded_pupil_signal = pa.read_pupil_data(pa.combined_h5_filename, signal_type='clean_signal')

	resampled_pupil_signal = sp.signal.resample(recorded_pupil_signal, int((recorded_pupil_signal.shape[-1] / signal_sample_frequency)*deconv_sample_frequency), axis = -1)

	trial_params = pa.read_trial_data(pa.combined_h5_filename)


		
	tc_rts = trial_params['reaction_time'][~np.isnan(trial_params['trial_phase_7_full_signal'])][:-1]

	#reg_stimulus_phase = trial_params['trial_phase_4_full_signal'][~np.isnan(trial_params['trial_phase_7_full_signal'])][:-1] / signal_sample_frequency*deconv_sample_frequency
	reg_response_phase = tc_rts * deconv_sample_frequency + trial_params['trial_phase_7_full_signal'][~np.isnan(trial_params['trial_phase_7_full_signal'])][:-1] / signal_sample_frequency*deconv_sample_frequency
	# reg_dec_interval = reg_response_phase
	# reg_response_phase_start = 
	# reg_button_press = reg_response_phase + tc_rts
	nr_trials = 0#reg_response_phase.shape[0]

	# tc_sigs = []
	# for trii,trial in enumerate(reg_response_phase):
	# 	# if (trial+(trial_deconvolution_interval[1]*deconv_sample_frequency)) <= resampled_pupil_signal.shape[0]:
	# 	# 	nr_trials += 1
	# 	sig = resampled_pupil_signal[trial+(trial_deconvolution_interval[0]*deconv_sample_frequency):trial+(trial_deconvolution_interval[1]*deconv_sample_frequency)]
	# 	sig -= sig[:int(0.5*deconv_sample_frequency)].mean()

	# 	tc_sigs.append(sig)
	# 	# else:
	# 	# 	tc_rts = tc_rts.delete(trii)
	# pupil_time_series = np.concatenate(tc_sigs)

	tstart = int(trial_deconvolution_interval[0]*deconv_sample_frequency)
	tend = int(trial_deconvolution_interval[1]*deconv_sample_frequency)
	bstart = int(0.5*deconv_sample_frequency)

	pupil_time_series = np.concatenate([resampled_pupil_signal[int(trial+tstart):int(trial+tend)]-resampled_pupil_signal[int(trial+tstart-bstart):int(trial+tstart)].mean() for trial in reg_response_phase])

	all_stim_events = np.cumsum(np.repeat(trial_deconvolution_interval[1]-trial_deconvolution_interval[0], reg_response_phase.size)) - trial_deconvolution_interval[1] - tc_rts
	all_resp_events = np.cumsum(np.repeat(trial_deconvolution_interval[1]-trial_deconvolution_interval[0], reg_response_phase.size)) - trial_deconvolution_interval[1]
	# all_button_events = all_resp_events + tc_rts.values

	tcodes = [0,10,30,50,70]
	tnames = ['noPE','PEtr','PEntr','bothPE']
	tcolors = ['k','r','g','b']
	# embed()

	#try:
	
	all_events = []
	all_event_types = []
	for tcii in range(len(tcodes)-1):

		trial_iis = np.array((trial_params['trial_codes'][~np.isnan(trial_params['trial_phase_7_full_signal'])][:-1] >= tcodes[tcii]) * (trial_params['trial_codes'][~np.isnan(trial_params['trial_phase_7_full_signal'])][:-1] < tcodes[tcii+1]), dtype=bool)

		event_cue = np.zeros((trial_iis.sum(), 3))
		event_decint = np.zeros((trial_iis.sum(), 3))
		event_button = np.zeros((trial_iis.sum(), 3))

		
		# reg_stimulus_phase_start = reg_response_phase_start - 0.33#np.cumsum(np.repeat(trial_deconvolution_interval[1]-trial_deconvolution_interval[0], nr_trials)) - trial_deconvolution_interval[1]
		# reg_button_press = reg_response_phase_start + tc_rts.values[:nr_trials]#*deconv_sample_frequency

		# event_cue[:,0] = reg_stimulus_phase_start
		# event_cue[:,1] = 0.33
		# event_cue[:,2] = 1

		event_cue[:,0] = all_stim_events[trial_iis]
		event_cue[:,1] = 0
		event_cue[:,2] = 1

		event_decint[:,0] = all_stim_events[trial_iis]
		event_decint[:,1] = tc_rts.values[trial_iis]# * deconv_sample_frequency
		event_decint[:,2] = 2

		event_button[:,0] = all_resp_events[trial_iis]
		event_button[:,1] = 0
		event_button[:,2] = 3
		# reg_response_phase_start = np.arange(trial_deconvolution_interval[0]*deconv_sample_frequency, nr_trials * , np.sum(trial_deconvolution_interval)*deconv_sample_frequency)

		all_events.extend([event_cue, event_decint, event_button])
		all_event_types.extend(['stick','box','stick'])
		
	linear_model = GLM(input_object=pupil_time_series, event_object=all_events, sample_dur=1/deconv_sample_frequency, new_sample_dur=1/deconv_sample_frequency)
	linear_model.configure(IRF='pupil', IRF_params={'dur':3, 's':1.0/(10**26), 'n':10.1, 'tmax':0.93}, regressor_types=all_event_types, normalize_sustained = True)
	linear_model.execute()

		
	plt.figure()
	
	main_sub_inds = [1, 3, 5, 7]
	inset_sub_inds = [2, 4, 6, 8]
	beta_inds = [np.arange(3),np.arange(3,6),np.arange(6,9),np.arange(9,12)]

	for tcii in range(len(tcodes)-1):

		plt.subplot(2,4,main_sub_inds[tcii])
		plt.title(tnames[tcii])

		trial_iis = np.array((trial_params['trial_codes'][~np.isnan(trial_params['trial_phase_7_full_signal'])][:-1] >= tcodes[tcii]) * (trial_params['trial_codes'][~np.isnan(trial_params['trial_phase_7_full_signal'])][:-1] < tcodes[tcii+1]), dtype=bool)	

		avg_time_series = pupil_time_series.reshape((int(all_resp_events.shape[0]), int(pupil_time_series.shape[0]/all_resp_events.shape[0])))[trial_iis,:]
		avg_pred_series = linear_model.predicted.reshape((int(all_resp_events.shape[0]), int(pupil_time_series.shape[0]/all_resp_events.shape[0])))[trial_iis,:]

		sn.tsplot(avg_time_series, condition = tnames[tcii], legend=False, color=tcolors[tcii], ls='solid', time = pd.Series(data=np.arange(trial_deconvolution_interval[0], trial_deconvolution_interval[1], 1/deconv_sample_frequency), name= 'Time(s)'))
		sn.tsplot(avg_pred_series, condition = tnames[tcii], legend=False, color=tcolors[tcii], ls='dashed', time = pd.Series(data=np.arange(trial_deconvolution_interval[0], trial_deconvolution_interval[1], 1/deconv_sample_frequency), name= 'Time(s)'))
	# plt.plot(np.arange(0,pupil_time_series.size), linear_model.residuals, color='r',alpha=1)

		sn.despine(offset=2)

		ax=plt.subplot(2,4,inset_sub_inds[tcii])

		plt.bar([0.5,1.5,2.5], linear_model.betas[beta_inds[tcii]])
		ax.set(xticks=[0.5,1.5,2.5],xticklabels=['stim','int','button'])
		sn.despine()



	betas = linear_model.betas
	# betas = [np.NaN if b < 0 else b for b in betas]
	all_betas['PP'].append(betas[0:3])
	all_betas['UP'].append(betas[3:6])
	all_betas['PU'].append(betas[6:9])
	all_betas['UU'].append(betas[9:])

	plt.tight_layout()

	plt.savefig(os.path.join(figfolder,'per_sub','GLM','%s-GLM.pdf'%subname))

	plt.close()

	# embed()

	# except:
	# 	embed()
	# tc_correlations = dict(zip(tnames,[[]]*4))
	# for tcii in range(len(tcodes)-1):
		
	# 	tc_rts = trial_params['reaction_time'][(trial_params['trial_codes'] >= tcodes[tcii]) * (trial_params['trial_codes'] < tcodes[tcii+1])]

	# 	trials = tc_rts + trial_params['trial_phase_7_full_signal'][(trial_params['trial_codes'] >= tcodes[tcii]) * (trial_params['trial_codes'] < tcodes[tcii+1])] / signal_sample_frequency*deconv_sample_frequency

	# 	tc_sigs = []
	# 	for trii,trial in enumerate(trials):
	# 		if (trial+(trial_deconvolution_interval[1]*deconv_sample_frequency)) <= resampled_pupil_signal.shape[0]:
	# 			tc_sigs.append(resampled_pupil_signal[trial+(trial_deconvolution_interval[0]*deconv_sample_frequency):trial+(trial_deconvolution_interval[1]*deconv_sample_frequency)])
	# 		# else:
	# 		# 	tc_rts = tc_rts.delete(trii)
	# 	tc_sigs = np.vstack(tc_sigs)
	# 	tc_rts = tc_rts[:tc_sigs.shape[0]]
	# 	tc_corr = []
	# 	for timep in range(tc_sigs.shape[1]):
	# 		tc_corr.append(np.corrcoef(tc_sigs[:,timep], tc_rts)[0][1])

	# 	tc_correlations[tnames[tcii]] = np.array(tc_corr)
	# 	all_correlations[['PP','UP','PU','UU'][tcii]].append(np.array(tc_corr))

	


	# embed()

# embed()

all_data_ndarray = np.dstack([all_betas['PU'],all_betas['PP'],all_betas['UU'],all_betas['UP']])

pd_data = pd.DataFrame(data=np.vstack([all_data_ndarray.ravel(), np.tile(['stim','decint','button'], all_data_ndarray.shape[0]*all_data_ndarray.shape[2]), np.tile(np.repeat(np.arange(0,all_data_ndarray.shape[0]), all_data_ndarray.shape[1]), all_data_ndarray.shape[2]), np.repeat(['PEntr','noPE','bothPE','PEtr'], all_data_ndarray.shape[0]*all_data_ndarray.shape[1])]).T,
					   index = np.arange(all_data_ndarray.shape[0]*all_data_ndarray.shape[1]*all_data_ndarray.shape[2]),
					   columns=['beta','param','pp','condition'])

pd_data['beta']=pd.to_numeric(pd_data['beta'],errors='coerce') 
pd_data['pp']=pd.to_numeric(pd_data['pp'],errors='coerce') 


plt.figure()

# plt.ylabel(r'Pupil-RT correlation ($r$)')
# plt.axvline(x=0, color='k', linestyle='solid', alpha=0.15)
# plt.axhline(y=0, color='k', linestyle='dashed', alpha=0.25)
sn.factorplot(data = pd_data.dropna(subset=['beta']), x = 'param', y='beta', hue = 'condition')


# sn.despine(5)
plt.savefig(os.path.join(figfolder,'over_subs','GLM_betas_all.pdf'))

plt.close()
# pl.open_figure(force=1)
# pl.hline(y=0)
# pl.vline(x=np.abs(trial_deconvolution_interval[0])*deconv_sample_frequency)
# pl.event_related_pupil_average(data = all_correlations, conditions = ['PP','UP','PU','UU'], show_legend = True, ylabel = 'Correlation (r)', signal_labels = dict(zip(['PU','PP','UU','UP'], tnames)), xticks = np.arange(0, all_correlations['PP'][0].shape[0], int(all_correlations['PP'][0].shape[0]/deconv_sample_frequency)), xticklabels = np.arange(trial_deconvolution_interval[0], trial_deconvolution_interval[1]+1, .5), compute_mean = True, compute_sd = True)
# pl.save_figure(filename='tc_corr_response.pdf',sub_folder = 'over_subs/RT')

