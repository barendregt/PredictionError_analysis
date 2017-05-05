import numpy as np
import scipy as sp

from math import *
import os,glob,sys

from joblib import Parallel, delayed
import multiprocessing

import cPickle as pickle
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn
sn.set(style='ticks')

from IPython import embed

# sys.path.append('tools/')
from BehaviorAnalyzer import BehaviorAnalyzer


raw_data_folder = '/home/raw_data/2017/visual/PredictionError/Behavioural/Reaction_times/'#'/home/barendregt/Projects/PredictionError/Psychophysics/Data/k1f46/' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/PredictionError/Figures'

sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO']#
# sublist = ['AO']
#['s1','s2','s3','s4','s5','s6']#['s1','s2','s4']['s1','s2',[

# subname = 's1'#'tk2'#'s3'#'mb2'#'tk2'#'s3'#'tk2'

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 8
deconvolution_interval = np.array([-0.5, 4.5])

down_fs = 10
winlength = 4500#6500
minwinlength = 4000#6000
 
def run_analysis(subname):	

	print '[main] Running analysis for %s' % (subname)

	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)

	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]

	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 3, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)

	pa.recombine_signal_blocks(force_rebuild = True)

	signal_types = ['gaze_x',
					'gaze_y',
					'pupil',
					'vel_x',
					'vel_y',
					'pupil_int',
					'pupil_hp',
					'pupil_lp',
					'pupil_lp_psc',
					'pupil_lp_diff',
					'pupil_bp',
					'pupil_bp_dt',
					'pupil_bp_zscore',
					'pupil_bp_psc',
					'pupil_baseline',
					'gaze_x_int',
					'gaze_y_int',
					'pupil_lp_clean',
					'pupil_lp_clean_psc',
					'pupil_lp_clean_zscore',
					'pupil_bp_clean',
					'pupil_bp_clean_psc',
					'pupil_bp_clean_zscore',
					'pupil_bp_clean_dt']

	selected_types = ['pupil_int',
					  'pupil_bp',
					  'pupil_bp_clean']

	pa.load_data()

	pa.get_aliases()

	#if not os.path.isdir(os.path.join(pa.edf_folder,'report/')):
	reference_phase = 1

	for ii,alias in enumerate(pa.aliases):

		if not os.path.isdir(os.path.join(pa.edf_folder,'report',alias + '/')):
			os.makedirs(os.path.join(pa.edf_folder,'report',alias + '/'))
		# os.mkdirs(os.path.join(pa.edf_folder,'report',alias,'per_run/'))

		blocks = pa.h5_operator.read_session_data(alias, 'blocks')
		block_start_times = blocks['block_start_timestamp']

		
		
		this_trial_phase_times = pa.h5_operator.read_session_data(alias, 'trial_phases')
		this_trial_phase_times = this_trial_phase_times[this_trial_phase_times['trial_phase_index']==reference_phase]
		this_phase_times = np.array(this_trial_phase_times['trial_phase_EL_timestamp'].values, dtype=float)
#
		for bs,be in zip(blocks['block_start_timestamp'], blocks['block_end_timestamp']):

			selected_signals = {}



			for sig_type in signal_types:
				this_block_signal = np.squeeze(pa.h5_operator.signal_during_period(time_period = (bs, be), alias = alias, signal = sig_type, requested_eye = pa.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))

			# 	plt.figure(figsize=(10,6))
			# 	plt.title(sig_type)
			# 	#plt.axvline(this_phase_times[(this_phase_times >= bs) & (this_phase_times < be)],0,1,alpha=0.5)
			# 	plt.plot(this_block_signal)
			# 	plt.savefig(os.path.join(pa.edf_folder,'report',alias,sig_type+'.jpg'))
			# 	plt.close()

				if sig_type in selected_types:
					selected_signals.update({sig_type: this_block_signal})

			plt.figure()

			for sii,stype in enumerate(selected_types):
				splt = plt.subplot(3,1,sii+1)

				splt.set_title(stype)

				for xpos in this_phase_times[(this_phase_times >= bs) & (this_phase_times < be)] - bs:
					splt.axvline(xpos,0,1,alpha=0.5)
				splt.plot(selected_signals[stype])
				sn.despine()
			plt.tight_layout()

			plt.savefig(os.path.join(pa.edf_folder,'report',alias + '_full_signal.jpg'))
			plt.close()

			plt.figure()

			trial_inds = this_phase_times[(this_phase_times >= bs) & (this_phase_times < be)] - bs

			plt.plot(np.mean([selected_signals['pupil_bp_clean'].values[int(tii):int(tii)+1000*6] for tii in trial_inds[:-1]], axis=0))

			plt.savefig(os.path.join(pa.edf_folder,'report',alias + '_signal.jpg'))
			plt.close()


			plt.figure()

			this_parameters = pa.read_trial_data(input_file = pa.combined_h5_filename, run = 'run%i'%ii)

			PP_rt = []
			PU_rt = []
			UP_rt = []
			UU_rt = []

			for trial_code in np.unique(this_parameters['trial_codes']):
				if trial_code < 10:
					PP_rt.extend(this_parameters['reaction_time'][(this_parameters['correct_answer']==1) *(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])
				elif trial_code < 30:
					PU_rt.extend(this_parameters['reaction_time'][(this_parameters['correct_answer']==1) *(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])
				elif trial_code < 50:
					UP_rt.extend(this_parameters['reaction_time'][(this_parameters['correct_answer']==1) *(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])
				else:
					UU_rt.extend(this_parameters['reaction_time'][(this_parameters['correct_answer']==1) *(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])

			splt = plt.subplot(2,2,1)

			splt.set_title('RT correct')

			splt.bar([0.5,1.5,2.5,3.5], [np.median(PP_rt), np.median(PU_rt), np.median(UP_rt), np.median(UU_rt)], color='w', edgecolor='k')

			splt.set(xticklabels = ['PP','PU','UP','UU'])

			PP_rt = []
			PU_rt = []
			UP_rt = []
			UU_rt = []

			for trial_code in np.unique(this_parameters['trial_codes']):
				if trial_code < 10:
					PP_rt.extend(this_parameters['reaction_time'][(this_parameters['correct_answer']==0) *(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])
				elif trial_code < 30:
					PU_rt.extend(this_parameters['reaction_time'][(this_parameters['correct_answer']==0) *(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])
				elif trial_code < 50:
					UP_rt.extend(this_parameters['reaction_time'][(this_parameters['correct_answer']==0) *(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])
				else:
					UU_rt.extend(this_parameters['reaction_time'][(this_parameters['correct_answer']==0) *(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])

			splt = plt.subplot(2,2,2)

			splt.set_title('RT incorrect')

			splt.bar([0.5,1.5,2.5,3.5], [np.median(PP_rt), np.median(PU_rt), np.median(UP_rt), np.median(UU_rt)], color='w', edgecolor='k')		

			splt.set(xticklabels = ['PP','PU','UP','UU'])

			PP_pc = []
			PU_pc = []
			UP_pc = []
			UU_pc = []

			for trial_code in np.unique(this_parameters['trial_codes']):
				if trial_code < 10:
					PP_pc.extend(this_parameters['correct_answer'][(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])
				elif trial_code < 30:
					PU_pc.extend(this_parameters['correct_answer'][(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])
				elif trial_code < 50:
					UP_pc.extend(this_parameters['correct_answer'][(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])
				else:
					UU_pc.extend(this_parameters['correct_answer'][(this_parameters['reaction_time']>0) * (this_parameters['trial_codes']==trial_code)])

			splt = plt.subplot(2,2,3)

			splt.set_title('Performance')

			splt.bar([0.5,1.5,2.5,3.5], [np.mean(PP_pc), np.mean(PU_pc), np.mean(UP_pc), np.mean(UU_pc)], color='w', edgecolor='k')		

			splt.set(xticklabels = ['PP','PU','UP','UU'])

			plt.tight_layout()

			plt.savefig(os.path.join(pa.edf_folder,'report',alias + '_behaviour.jpg'))

			plt.close()


	

	pa.signal_per_trial(only_correct = True, reference_phase = 1, with_rt = False, baseline_type = None, baseline_period = [-2.5, 0.0], force_rebuild=False)

	response_pupil_signals = []

	for key, trial_signal in pa.trial_signals.items():
		response_pupil_signals.extend(trial_signal)

	plt.figure()

	plt.plot(np.mean(response_pupil_signals, axis=0))

	sn.despine()

	plt.savefig(os.path.join(pa.edf_folder,'report','all_average_response.jpg'))

	plt.close()

	pa.unload_data()

#[run_analysis(sub) for sub in sublist]
#Run everything in parallel for speed 
# num_cores = multiprocessing.cpu_count()

Parallel(n_jobs=8)(delayed(run_analysis)(subname) for subname in sublist)
