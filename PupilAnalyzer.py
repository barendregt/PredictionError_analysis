# from __future__ import division

import os,glob,datetime

import numpy as np
import scipy as sp
import seaborn as sn
import matplotlib.pylab as plt
import pickle as pickle
import pandas as pd
import tables

from scipy.signal import fftconvolve, resample

from math import *

from hedfpy.EDFOperator import EDFOperator
from hedfpy.HDFEyeOperator import HDFEyeOperator
from hedfpy.EyeSignalOperator import EyeSignalOperator

from fir import FIRDeconvolution

from IPython import embed

from Analyzer import Analyzer

from PupilProcessor import PupilProcessor

sn.set(style = 'ticks')

class PupilAnalyzer(Analyzer):

	def __init__(self, subID, filename, edf_folder, sort_by_date = False, reference_phase = 7,verbosity = 0, **kwargs):

		
		self.PR = PupilProcessor(subID, filename, edf_folder, sort_by_date, verbosity, **kwargs)

		
		self.FIR_object 	= None
		self.pupil_signal 	= None
		self.events 		= {}
		self.pupil_data     = None


	def signal_per_trial(self, reference_phase = 1, only_correct = False, only_incorrect = False, only_FA = False, only_Hit = False, sdt_dir = 1, return_dt = False, return_rt = False, return_blinks = False, with_rt = False, baseline_phase = 1, baseline_correction = True, baseline_type = 'absolute', baseline_period = [-0.5, 0.0], force_rebuild = False, signal_type = 'clean_signal', down_sample = False):

		if only_correct==True and only_incorrect==True:
			display('Error: incompatible trial selection!!')
			return

		self.PR.load_combined_data(force_rebuild=force_rebuild)

		recorded_pupil_signal = self.PR.read_pupil_data(self.PR.combined_h5_filename, signal_type = signal_type)
		trial_parameters = self.PR.read_trial_data(self.PR.combined_h5_filename)

		self.trial_signals = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		if return_rt:
			self.trial_rts = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		# if return_blinks:
		# 	blinks = self.read_blink_data(self.PR.combined_h5_filename)
		# 	self.trial_blinks = {key:[] for key in np.unique(trial_parameters['trial_codes'])}


		for tcode in np.unique(trial_parameters['trial_codes']):
		

			if only_correct:
				selected_trials = np.array((trial_parameters['trial_codes']==tcode) & (trial_parameters['correct_answer']==1), dtype=bool)
			elif only_incorrect:
				selected_trials = np.array((trial_parameters['trial_codes']==tcode) & (trial_parameters['correct_answer']==0), dtype=bool)
			elif only_Hit:
				selected_trials = np.array((trial_parameters['trial_codes']==tcode) & ((trial_parameters['trial_direction']==sdt_dir) & (trial_parameters['response']==sdt_dir)), dtype=bool)		
			elif only_FA:
				selected_trials = np.array((trial_parameters['trial_codes']==tcode) & ((trial_parameters['trial_direction']==(-1*sdt_dir)) & (trial_parameters['response']==sdt_dir)), dtype=bool)						
			else:
				selected_trials = np.array(trial_parameters['trial_codes']==tcode, dtype=bool)			


			if with_rt:
				trial_times = list(zip(trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + (trial_parameters['reaction_time'][selected_trials].values*self.PR.signal_sample_frequency) + ((self.PR.deconvolution_interval)*self.PR.signal_sample_frequency)[0], trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + (trial_parameters['reaction_time'][selected_trials].values*self.PR.signal_sample_frequency) + ((self.PR.deconvolution_interval)*self.PR.signal_sample_frequency)[1]))
			else:
				trial_times = list(zip(trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + ((self.PR.deconvolution_interval)*self.PR.signal_sample_frequency)[0], trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + ((self.PR.deconvolution_interval)*self.PR.signal_sample_frequency)[1]))

			if return_rt:
				these_rts = trial_parameters['reaction_time'][selected_trials].values

			if baseline_correction:
				if baseline_type == 'relative':
					baseline_times =  np.vstack([trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + (baseline_period[0]*self.PR.signal_sample_frequency), trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + (baseline_period[1]*self.PR.signal_sample_frequency)])
				else:
					baseline_times =  np.vstack([trial_parameters['trial_phase_%i_full_signal'%baseline_phase][selected_trials].values + (baseline_period[0]*self.PR.signal_sample_frequency), trial_parameters['trial_phase_%i_full_signal'%baseline_phase][selected_trials].values + (baseline_period[1]*self.PR.signal_sample_frequency)])

			for tii,(ts,te) in enumerate(trial_times):
				if (ts > 0) & (te < recorded_pupil_signal.size):

					trial_pupil_signal = recorded_pupil_signal[int(ts):int(te)]



					if baseline_correction:
						trial_pupil_signal -= np.mean(recorded_pupil_signal[int(baseline_times[0, int(tii)]):int(baseline_times[1,int(tii)])])

						# trial_pupil_signal /= np.mean(trial_pupil_signal)


					# sp.signal.decimate(trial_pupil_signal, self.signal_downsample_factor, 8))?

					# self.trial_signals[tcode].append(resample(trial_pupil_signal, round(len(trial_pupil_signal)/self.signal_downsample_factor)))
					if down_sample:
						trial_pupil_signal = sp.signal.decimate(trial_pupil_signal, self.PR.signal_downsample_factor, 1)

					if return_dt:
						trial_pupil_signal = np.hstack([0, np.diff(trial_pupil_signal)])

					self.trial_signals[tcode].append(trial_pupil_signal)

					if return_rt:
						self.trial_rts[tcode].append(these_rts[tii])

					# if return_blinks:
					# 	self.trial_blinks[tcode].append(blinks[(blinks['start_timestamp']/self.PR.signal_sample_frequency >= ts) * (blinks['end_timestamp']/self.PR.signal_sample_frequency < te)])

					
			self.trial_signals[tcode] = np.array(self.trial_signals[tcode])
			if return_rt:
				self.trial_rts[tcode] = np.array(self.trial_rts[tcode])

	def compute_TPR(self, reference_phase = 1, only_correct = False, only_incorrect = False, time_window = None, baseline_period = [-0.5, 0.0], with_rt = False, force_rebuild = False, signal_type = 'clean_signal', down_sample = False, sort_by_code = True):

		"""
		Compute a trial-by-trial pupil response, sorted by trial code
		"""

		if only_correct==True and only_incorrect==True:
			display('Error: incompatible trial selection!!')
			# embed()
			return

		trial_start_offset = 0

		if time_window is None:
			time_window = self.PR.deconvolution_interval

		self.PR.load_combined_data(force_rebuild=force_rebuild)

		recorded_pupil_signal = self.PR.read_pupil_data(self.PR.combined_h5_filename, signal_type = signal_type)
		trial_parameters = self.PR.read_trial_data(self.PR.combined_h5_filename)

		if sort_by_code:
			self.TPR =  {key:[] for key in np.unique(trial_parameters['trial_codes'])}
		else:
			self.TPR = []


		#for tcode in np.unique(trial_parameters['trial_codes']):
		

		if only_correct:
			selected_trials = np.array(trial_parameters['correct_answer']==1, dtype=bool)
		elif only_incorrect:
			selected_trials = np.array(trial_parameters['correct_answer']==0, dtype=bool)				
		else:
			selected_trials = np.array((trial_parameters['correct_answer']==1) | (trial_parameters['correct_answer']==0), dtype=bool)			


		if with_rt:
			trial_times = list(zip(trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + (trial_parameters['reaction_time'][selected_trials].values*self.PR.signal_sample_frequency) + ((self.PR.deconvolution_interval)*self.PR.signal_sample_frequency)[0], trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + (trial_parameters['reaction_time'][selected_trials].values*self.PR.signal_sample_frequency) + ((self.PR.deconvolution_interval)*self.PR.signal_sample_frequency)[1]))
		else:
			trial_times = list(zip(trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + ((time_window)*self.PR.signal_sample_frequency)[0], trial_parameters['trial_phase_%i_full_signal'%reference_phase][selected_trials].values + ((time_window)*self.PR.signal_sample_frequency)[1]))


		baseline_times =  np.vstack([trial_parameters['trial_phase_1_full_signal'][selected_trials].values + (baseline_period[0]*self.PR.signal_sample_frequency), trial_parameters['trial_phase_1_full_signal'][selected_trials].values + (baseline_period[1]*self.PR.signal_sample_frequency)])

		for tii,(ts,te) in enumerate(trial_times):
			if (ts > 0) & (te < recorded_pupil_signal.size):

				trial_pupil_response = recorded_pupil_signal[int(ts):int(te)] - np.mean(recorded_pupil_signal[int(baseline_times[0, int(tii)]):int(baseline_times[1,int(tii)])])
				trial_pupil_response = np.max(np.abs(trial_pupil_response))
				# trial_puresponsegnal = np.mean(recorded_pupil_signal[int(ts):int(te)]) - 

				if sort_by_code:
					self.TPR[trial_parameters['trial_codes'][tii]].append(trial_pupil_response)
				else:
					self.TPR.append(trial_pupil_response)
					
		if sort_by_code:

			for tcode in np.unique(trial_parameters['trial_codes']):
				self.TPR[tcode] = np.array(self.TPR[tcode])
		else:
			self.TPR = np.array(self.TPR)



	def get_IRF(self, deconv_interval = None, only_correct = False):

		self.PR.load_combined_data()
		# embed()

		recorded_pupil_signal = self.read_pupil_data(self.PR.combined_h5_filename, signal_type = 'long_signal')

		self.FIR_resampled_pupil_signal = sp.signal.resample(recorded_pupil_signal, int((recorded_pupil_signal.shape[-1] / self.PR.signal_sample_frequency)*self.deconv_sample_frequency), axis = -1)


		trial_parameters = self.read_trial_data(self.PR.combined_h5_filename)
		blinks = self.read_blink_data(self.PR.combined_h5_filename)
		saccades = self.read_saccade_data(self.PR.combined_h5_filename)

		nuiss_events = np.array([blinks['end_block_timestamp'],
							   saccades['end_block_timestamp']])#,
							   #trial_parameters['trial_phase_2_full_signal']])#,   # task cue
							   #(trial_parameters['reaction_time'][trial_parameters['trial_stimulus']<2]*self.PR.signal_sample_frequency)+trial_parameters['trial_phase_4_full_signal'][(~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_stimulus']<2],   # red stimulus
					  		   #(trial_parameters['reaction_time'][trial_parameters['trial_stimulus']>=2]*self.PR.signal_sample_frequency)+trial_parameters['trial_phase_4_full_signal'][trial_parameters['trial_stimulus']>=2]]) # green stimulus

		#if deconv_interval is None:
		nuiss_deconv_interval = [-2, 5]


		print(('[%s] Starting FIR deconvolution' % (self.__class__.__name__)))

		self.FIR_nuiss = FIRDeconvolution(
						signal = recorded_pupil_signal,
						events = nuiss_events / self.PR.signal_sample_frequency,
						event_names = ['blinks','saccades'],#,'task_cue'],#,'red_stim','green_stim'],
						#durations = {'response': self.events['durations']['response']},
						sample_frequency = self.PR.signal_sample_frequency,
			            deconvolution_frequency = self.deconv_sample_frequency,
			        	deconvolution_interval = nuiss_deconv_interval,
			        	#covariates = self.events['covariates']
					)

		self.FIR_nuiss.create_design_matrix(intercept=False)
		# dm_nuiss = self.FIR_nuiss.design_matrix

		try:
			# One stimulus-locked, color tas

			stim_deconv_interval = [-0.5,3]

			if only_correct:
				stim_events = np.array([trial_parameters['trial_phase_4_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_codes'] < 10)], # no PE
								   trial_parameters['trial_phase_4_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_codes'] >= 50)], # both PE
								   trial_parameters['trial_phase_4_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_codes'] >= 10) * (trial_parameters['trial_codes'] < 30)], # PE TR
								   trial_parameters['trial_phase_4_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_codes'] >= 30) * (trial_parameters['trial_codes'] < 50)]  # PE ~TR
								  ])
			else:
				stim_events = np.array([trial_parameters['trial_phase_4_full_signal'][(~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_codes'] < 10)], # no PE
								   trial_parameters['trial_phase_4_full_signal'][(~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_codes'] >= 50)], # both PE
								   trial_parameters['trial_phase_4_full_signal'][(~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_codes'] >= 10) * (trial_parameters['trial_codes'] < 30)], # PE TR
								   trial_parameters['trial_phase_4_full_signal'][(~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_codes'] >= 30) * (trial_parameters['trial_codes'] < 50)]  # PE ~TR
								  ])

			# covariates = {''}

			self.FIR_stim = FIRDeconvolution(
							signal = recorded_pupil_signal,
							events = stim_events / self.PR.signal_sample_frequency,
							event_names = ['noPE','bothPE','PEtr','PEntr'],
							#durations = {'response': self.events['durations']['response']},
							sample_frequency = self.PR.signal_sample_frequency,
				            deconvolution_frequency = self.deconv_sample_frequency,
				        	deconvolution_interval = stim_deconv_interval,
				        	#covariates = self.events['covariates']
						)

			self.FIR_stim.create_design_matrix(intercept=False)

			# dm_stim_color = self.FIR_stim_color.design_matrix

			# One response-locked

			resp_deconv_interval = [-2,3]


			if only_correct:
				resp_events = np.array([(trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['reaction_time'])) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes'] < 10)] * self.PR.signal_sample_frequency) + trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['reaction_time'])) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes'] < 10)], # no PE
								   (trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['reaction_time'])) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes'] >= 50)] * self.PR.signal_sample_frequency) + trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['reaction_time'])) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes'] >= 50)], # both PE
								   (trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['reaction_time'])) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes'] >= 10) * (trial_parameters['trial_codes'] < 30)] * self.PR.signal_sample_frequency) + trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['reaction_time'])) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes'] >= 10) * (trial_parameters['trial_codes'] < 30)], # PE TR
								   (trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['reaction_time'])) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes'] >= 30) * (trial_parameters['trial_codes'] < 50)] * self.PR.signal_sample_frequency) + trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['reaction_time'])) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes'] >= 30) * (trial_parameters['trial_codes'] < 50)]  # PE ~TR
								  ])
			else:
				resp_events = np.array([(trial_parameters['reaction_time'][(~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (~np.isnan(trial_parameters['reaction_time'])) * (trial_parameters['trial_codes'] < 10)] * self.PR.signal_sample_frequency) + trial_parameters['trial_phase_7_full_signal'][(~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (~np.isnan(trial_parameters['reaction_time'])) * (trial_parameters['trial_codes'] < 10)], # no PE
								  (trial_parameters['reaction_time'][(~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (~np.isnan(trial_parameters['reaction_time'])) * (trial_parameters['trial_codes'] >= 50)] * self.PR.signal_sample_frequency) + trial_parameters['trial_phase_7_full_signal'][(~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (~np.isnan(trial_parameters['reaction_time'])) * (trial_parameters['trial_codes'] >= 50)], # both PE
								  (trial_parameters['reaction_time'][(~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (~np.isnan(trial_parameters['reaction_time'])) * (trial_parameters['trial_codes'] >= 10) * (trial_parameters['trial_codes'] < 30)] * self.PR.signal_sample_frequency) + trial_parameters['trial_phase_7_full_signal'][(~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (~np.isnan(trial_parameters['reaction_time'])) * (trial_parameters['trial_codes'] >= 10) * (trial_parameters['trial_codes'] < 30)], # PE TR
								  (trial_parameters['reaction_time'][(~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (~np.isnan(trial_parameters['reaction_time'])) * (trial_parameters['trial_codes'] >= 30) * (trial_parameters['trial_codes'] < 50)] * self.PR.signal_sample_frequency) + trial_parameters['trial_phase_7_full_signal'][(~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (~np.isnan(trial_parameters['reaction_time'])) * (trial_parameters['trial_codes'] >= 30) * (trial_parameters['trial_codes'] < 50)]  # PE ~TR
				 				  ])


			self.FIR_resp = FIRDeconvolution(
							signal = recorded_pupil_signal,
							events = resp_events / self.PR.signal_sample_frequency,
							event_names = ['noPE','bothPE','PEtr','PEntr'],
							#durations = {'response': self.events['durations']['response']},
							sample_frequency = self.PR.signal_sample_frequency,
				            deconvolution_frequency = self.deconv_sample_frequency,
				        	deconvolution_interval = resp_deconv_interval,
				        	#covariates = self.events['covariates']
						)

			self.FIR_resp.create_design_matrix(intercept=False)

		except:
			print('Well, that didnt work...')
			

		# dm3 = self.FIR_resp_ori.design_matrix		


		self.dm_stim = np.vstack([self.FIR_nuiss.design_matrix, self.FIR_stim.design_matrix])

	
		self.FIR_betas_stim = sp.linalg.lstsq(self.dm_stim.T, self.FIR_resampled_pupil_signal.T)[0]

		self.dm_resp = np.vstack([self.FIR_nuiss.design_matrix, self.FIR_resp.design_matrix])

		self.FIR_betas_resp = sp.linalg.lstsq(self.dm_resp.T, self.FIR_resampled_pupil_signal.T)[0]


		self.dm_all = np.vstack([self.FIR_nuiss.design_matrix, self.FIR_stim.design_matrix, self.FIR_resp.design_matrix])

		self.FIR_betas_all = sp.linalg.lstsq(self.dm_all.T, self.FIR_resampled_pupil_signal.T)[0]	
		
		# embed()
		stim_betas = self.FIR_betas_stim[-int(stim_events.shape[0]*(stim_deconv_interval[1]-stim_deconv_interval[0])*self.deconv_sample_frequency):].reshape((stim_events.shape[0],int((stim_deconv_interval[1]-stim_deconv_interval[0])*self.deconv_sample_frequency))).T

		stim_nuiss_betas = self.FIR_betas_stim[:-int(stim_events.shape[0]*(stim_deconv_interval[1]-stim_deconv_interval[0])*self.deconv_sample_frequency)].reshape((nuiss_events.shape[0],(nuiss_deconv_interval[1]-nuiss_deconv_interval[0])*self.deconv_sample_frequency)).T


		resp_betas = self.FIR_betas_resp[-int(resp_events.shape[0]*(resp_deconv_interval[1]-resp_deconv_interval[0])*self.deconv_sample_frequency):].reshape((resp_events.shape[0],(resp_deconv_interval[1]-resp_deconv_interval[0])*self.deconv_sample_frequency)).T
		resp_nuiss_betas = self.FIR_betas_resp[:-int(resp_events.shape[0]*(resp_deconv_interval[1]-resp_deconv_interval[0])*self.deconv_sample_frequency)].reshape((nuiss_events.shape[0],(nuiss_deconv_interval[1]-nuiss_deconv_interval[0])*self.deconv_sample_frequency)).T


		# all_betas = self.FIR_betas_all[-int(stim_events.shape[0]*(stim_deconv_interval[1]-stim_deconv_interval[0])*self.deconv_sample_frequency+resp_events.shape[0]*(resp_deconv_interval[1]-resp_deconv_interval[0])*self.deconv_sample_frequency):].reshape((stim_events.shape[0]+resp_events.shape[0],((stim_deconv_interval[1]-stim_deconv_interval[0])+(resp_deconv_interval[1]-resp_deconv_interval[0]))*self.deconv_sample_frequency)).T
		# all_stim_betas = afll_betas[:,0:4]
		# all_resp_betas = all_betas[:,4:]
		# all_nuiss_betas = self.FIR_betas_all[:-int(stim_events.shape[0]*(stim_deconv_interval[1]-stim_deconv_interval[0])*self.deconv_sample_frequency+resp_events.shape[0]*(resp_deconv_interval[1]-resp_deconv_interval[0])*self.deconv_sample_frequency)].reshape((nuiss_events.shape[0],(nuiss_deconv_interval[1]-nuiss_deconv_interval[0])*self.deconv_sample_frequency)).T

		return [[stim_betas, resp_betas, stim_nuiss_betas, resp_nuiss_betas], [list(self.FIR_stim.covariates.keys()), list(self.FIR_resp.covariates.keys()),list(self.FIR_nuiss.covariates.keys())]]

		# figure()
		# ax=subplot(1,2,1)
		# title('Nuissances')
		# plot(other_betas)
		# legend(self.FIR1.covariates.keys())

		# ax.set(xticks=np.arange(0,100,10), xticklabels=np.arange(-2,8))

		# sn.despine()

		# ax=subplot(1,2,2)
		# title('PE')
		# plot(pe_betas)
		# legend(self.FIR2.covariates.keys())
		# ax.set(xticks=np.arange(0,40,10), xticklabels=np.arange(-1,3))

		# sn.despine()

		# tight_layout()

		# savefig('fir_example.pdf')


		# print '[%s] Fitting IRF' % (self.__class__.__name__)
		# self.FIRo.regress(method = 'lstsq')
		# self.FIRo.betas_for_events()
		# self.FIRo.calculate_rsq()	

		# self.sub_IRF = {'stimulus': [], 'button_press': []}

		# self.sub_IRF['stimulus'] = self.FIRo.betas_per_event_type[0].ravel() - self.FIRo.betas_per_event_type[0].ravel().mean()
		# self.sub_IRF['button_press'] = self.FIRo.betas_per_event_type[1].ravel() - self.FIRo.betas_per_event_type[1].ravel().mean()

	def microsaccades_per_run(self, block_length = 5):
		

		self.get_aliases()

		block_length_samples = block_length * self.signal_downsample_factor

		
		ms_per_run = []
		signal_per_run = []
		for run_ii in range(len(self.aliases)):
			saccades = self.read_saccade_data(self.PR.combined_h5_filename, run = 'r%i_saccades'%run_ii)

			signal = self.read_pupil_data(self.PR.combined_h5_filename, signal_type = 'r%i_signal'%run_ii)

			block_times = np.arange(0, signal.size, block_length_samples)

			ms_per_block = np.array([])
			pupil_per_block = np.array([])

			for t in range(1,len(block_times)):
				ms_per_block = np.append(ms_per_block, np.sum(((saccades['start_block_timestamp']/self.PR.signal_sample_frequency*self.signal_downsample_factor) >= block_times[t-1]) * ((saccades['start_block_timestamp']/self.PR.signal_sample_frequency*self.signal_downsample_factor) < block_times[t]) * (saccades['length'] < 1.0)))
				pupil_per_block = np.append(pupil_per_block, np.mean(signal[block_times[t-1]:block_times[t]]))

			ms_per_run.append(ms_per_block)
			signal_per_run.append(pupil_per_block)

		return  ms_per_run, signal_per_run


	def compute_blink_rate(self, block_length = 5, max_blink_duration = 100000):


		self.get_aliases()

		block_length_samples = block_length * self.signal_downsample_factor
		


		blinks_per_run = []
		signal_per_run = []
		for run_ii in range(len(self.aliases)):

			trial_parameters = self.read_trial_data(self.PR.combined_h5_filename, run = '')

			blinks = self.read_blink_data(self.PR.combined_h5_filename, run = 'r%i_saccades'%run_ii)

			signal = self.read_pupil_data(self.PR.combined_h5_filename, signal_type = 'r%i_signal'%run_ii)

			block_times = np.arange(0, signal.size, block_length_samples)

			blinks_per_block = np.array([])
			pupil_per_block = np.array([])

			for t in range(1,len(block_times)):
				blinks_per_block = np.append(blinks_per_block, np.sum(((blinks['start_block_timestamp']/self.PR.signal_sample_frequency*self.signal_downsample_factor) >= block_times[t-1]) * ((blinks['start_block_timestamp']/self.PR.signal_sample_frequency*self.signal_downsample_factor) < block_times[t]) * (blinks['duration'] < max_blink_duration)))
				pupil_per_block = np.append(pupil_per_block, np.mean(signal[block_times[t-1]:block_times[t]]))

			blinks_per_run.append(blinks_per_block)
			signal_per_run.append(pupil_per_block)

		return  blinks_per_run, signal_per_run


	def get_IRF_correct_incorrect(self, deconv_interval = None, only_correct = False):

		self.PR.load_combined_data()


		recorded_pupil_signal = self.read_pupil_data(self.PR.combined_h5_filename, signal_type = 'long_signal')

		self.FIR_resampled_pupil_signal = sp.signal.resample(recorded_pupil_signal, int((recorded_pupil_signal.shape[-1] / self.PR.signal_sample_frequency)*self.deconv_sample_frequency), axis = -1)


		trial_parameters = self.read_trial_data(self.PR.combined_h5_filename)
		blinks = self.read_blink_data(self.PR.combined_h5_filename)
		saccades = self.read_saccade_data(self.PR.combined_h5_filename)

		nuiss_events = np.array([blinks['end_block_timestamp'],
							   saccades['end_block_timestamp']])#,
							   #trial_parameters['trial_phase_2_full_signal']])#,   # task cue
							   #(trial_parameters['reaction_time'][trial_parameters['trial_stimulus']<2]*self.PR.signal_sample_frequency)+trial_parameters['trial_phase_4_full_signal'][(~np.isnan(trial_parameters['trial_phase_4_full_signal'])) * (trial_parameters['trial_stimulus']<2],   # red stimulus
					  		   #(trial_parameters['reaction_time'][trial_parameters['trial_stimulus']>=2]*self.PR.signal_sample_frequency)+trial_parameters['trial_phase_4_full_signal'][trial_parameters['trial_stimulus']>=2]]) # green stimulus

		#if deconv_interval is None:
		nuiss_deconv_interval = [-2, 5]


		print(('[%s] Starting FIR deconvolution' % (self.__class__.__name__)))

		self.FIR_nuiss = FIRDeconvolution(
						signal = recorded_pupil_signal,
						events = nuiss_events / self.PR.signal_sample_frequency,
						event_names = ['blinks','saccades'],#,'task_cue'],#,'red_stim','green_stim'],
						#durations = {'response': self.events['durations']['response']},
						sample_frequency = self.PR.signal_sample_frequency,
			            deconvolution_frequency = self.deconv_sample_frequency,
			        	deconvolution_interval = nuiss_deconv_interval,
			        	#covariates = self.events['covariates']
					)

		self.FIR_nuiss.create_design_matrix(intercept=False)
		# dm_nuiss = self.FIR_nuiss.design_matrix

		try:
			# One stimulus-locked, color tas

			stim_deconv_interval = [-0.5,3]


			stim_events_correct = np.array([trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==0)], # no PE
							   trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==50)], # both PE
							   trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==10)], # PE ~TR
							   trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==30)]  # PE TR
							  ])

			stim_events_incorrect = np.array([trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==0) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==0)], # no PE
							   trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==0) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==50)], # both PE
							   trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==0) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==10)], # PE ~TR
							   trial_parameters['trial_phase_7_full_signal'][(trial_parameters['correct_answer']==0) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==30)]  # PE TR
							  ])

			stim_durs_correct = np.array([trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==0)], # no PE
							   trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==50)], # both PE
							   trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==10)], # PE ~TR
							   trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==30)]  # PE TR
							  ])

			stim_durs_incorrect = np.array([trial_parameters['reaction_time'][(trial_parameters['correct_answer']==0) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==0)], # no PE
							   trial_parameters['reaction_time'][(trial_parameters['correct_answer']==0) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==50)], # both PE
							   trial_parameters['reaction_time'][(trial_parameters['correct_answer']==0) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==10)], # PE ~TR
							   trial_parameters['reaction_time'][(trial_parameters['correct_answer']==0) * (~np.isnan(trial_parameters['trial_phase_7_full_signal'])) * (trial_parameters['trial_codes']==30)]  # PE TR
							  ])

			# covariates = {''}

			self.FIR = FIRDeconvolution(
							signal = recorded_pupil_signal,
							events = np.hstack([stim_events_correct / self.PR.signal_sample_frequency, stim_events_incorrect / self.PR.signal_sample_frequency]),
							event_names = ['noPEc','bothPEc','TIc','TRc','noPEic','bothPEic','TIic','TRic'],
							durations = {'noPEc': stim_durs_correct[0],
										 'bothPEc': stim_durs_correct[1],
										 'TRc': stim_durs_correct[3],
										 'TIc': stim_durs_correct[2],
										 'noPEic': stim_durs_incorrect[0],
										 'bothPEic': stim_durs_incorrect[1],
										 'TRic': stim_durs_incorrect[3],
										 'TIic': stim_durs_incorrect[2]},
							sample_frequency = self.PR.signal_sample_frequency,
				            deconvolution_frequency = self.deconv_sample_frequency,
				        	deconvolution_interval = stim_deconv_interval,
				        	#covariates = self.events['covariates']
						)

			self.FIR.create_design_matrix(intercept=False)

			# dm_stim_color = self.FIR_stim_color.design_matrix
		except:
			print('error!')
			embed()	
		self.dm_stim = np.vstack([self.FIR.design_matrix, self.FIR_nuiss.design_matrix])

	
		self.FIR_betas = sp.linalg.lstsq(self.dm_stim.T, self.FIR_resampled_pupil_signal.T)[0]


		stim_betas = np.reshape(self.FIR_betas[:280],(8,35))
		return [stim_betas, ['noPEc','bothPEc','TIc','TRc','noPEic','bothPEic','TIic','TRic']]


	def build_design_matrix(self, sub_IRF = None):
		
		print(('[%s] Creating design matrix for GLM' % (self.__class__.__name__)))

		if sub_IRF is None:
			sub_IRF = {'stimulus': [], 'button_press': []}

		

			for name,dec in zip(list(self.FIRo.covariates.keys()), self.FIRo.betas_per_event_type.squeeze()):
				sub_IRF[name] = resample(dec,int(len(dec)*(self.PR.signal_sample_frequency/self.deconv_sample_frequency)))[:,np.newaxis]
				sub_IRF[name] /= max(abs(sub_IRF[name]))


		self.PR.load_combined_data()

		recorded_pupil_signal = self.read_pupil_data(self.PR.combined_h5_filename, signal_type = 'long_signal')
		trial_parameters = self.read_trial_data(self.PR.combined_h5_filename)

		stim_times = trial_parameters['trial_response_phase_full_signal'] + (500+150+30+150)
		resp_times = trial_parameters['trial_response_phase_full_signal'] + self.PR.signal_sample_frequency*trial_parameters['reaction_time']

		# embed()

		self.design_matrix = np.ones((recorded_pupil_signal.size,1))	
		tempX = np.zeros((recorded_pupil_signal.size,1))

		for tcode in np.unique(trial_parameters['trial_codes']):
			for trial in np.where(trial_parameters['trial_codes']==tcode):
			# tempX[stim_times[] = 1


				self.design_matrix = np.hstack([self.design_matrix, fftconvolve(tempX, sub_IRF['stimulus'])[:recorded_pupil_signal.size], fftconvolve(tempX, sub_IRF['stimulus'])[:recorded_pupil_signal.size]])


	def run_GLM(self):

		print(('[%s] Running GLM analysis' % (self.__class__.__name__)))

		if not hasattr(self, 'design_matrix'):
			self.build_design_matrix()

		self.PR.load_combined_data()

		recorded_pupil_signal = self.read_pupil_data(self.PR.combined_h5_filename, signal_type = 'long_signal')
		trial_parameters = self.read_trial_data(self.PR.combined_h5_filename)

		trial_betas = np.linalg.pinv(self.design_matrix).dot(recorded_pupil_signal)