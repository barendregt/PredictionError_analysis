from __future__ import division
import os,glob,datetime

import numpy as np
import scipy as sp

# import cPickle as pickle
import pickle
import pandas as pd

from math import *

from hedfpy.EDFOperator import EDFOperator
from hedfpy.HDFEyeOperator import HDFEyeOperator
from hedfpy.EyeSignalOperator import EyeSignalOperator

from fir import FIRDeconvolution

from IPython import embed

from PupilAnalyzer import PupilAnalyzer

class BehaviorAnalyzer(PupilAnalyzer):

	def __init__(self, subID, csv_filename, h5_filename, raw_folder, verbosity=0, **kwargs):

		self.default_parameters = {}

		super(BehaviorAnalyzer, self).__init__(subID, h5_filename, raw_folder,verbosity=verbosity, **kwargs)

		self.csv_file = csv_filename

		self.h5_operator = None
		self.task_data = {}

		self.task_performance = {}

	def load_data(self):

		super(BehaviorAnalyzer, self).load_data()

		if len(self.csv_file)==1:
			self.csv_data = pd.read_csv(self.csv_file[0])
		else:
			self.csv_data_runs = []
			for ii in range(len(self.csv_file)):
				if ii>0:
					self.csv_data = pd.concat([self.csv_data, pd.read_csv(self.csv_file[ii], dtype={'trial_cue':'str', 'trial_stimulus_label': 'str'})], axis=0, ignore_index = True)
					self.csv_data_runs.append(pd.read_csv(self.csv_file[ii], dtype={'trial_cue':'str', 'trial_stimulus_label': 'str'}))
				else:
					self.csv_data = pd.read_csv(self.csv_file[ii], dtype={'trial_cue':'str', 'trial_stimulus_label': 'str'})
					self.csv_data_runs.append(pd.read_csv(self.csv_file[ii], dtype={'trial_cue':'str', 'trial_stimulus_label': 'str'}))


		if ('trial_cue' in self.csv_data.keys()) and ('trial_cue_label' not in self.csv_data.keys()):
			stimulus_types = {'red45': 0,
							  'red135': 1,
							  'green45': 2,
							  'green135': 3}

			self.csv_data['trial_cue_label'] = self.csv_data['trial_cue']
			self.csv_data['trial_cue'] = np.array([stimulus_types[tc] for tc in self.csv_data['trial_cue_label']], dtype=int)

		

	def recode_trial_types(self):
		"""
		Provide a simple coding scheme to extract trial according to 
		the type of stimulus presented:
			0:	base stimulus
			1:	change in attended feature
			2:	change in unattended feature
			3: 	change in both features
		"""

		self.load_data()

		self.get_aliases()

		new_trial_types = []

		trial_tasks = []
		trial_color = []
		trial_orientation = []
		trial_correct = []
		reaction_time = []

		for alias in self.aliases:

			trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

			trial_phase_times = self.h5_operator.read_session_data(alias, 'trial_phases')
			trial_times = self.h5_operator.read_session_data(alias, 'trials')
			#trial_types = self.h5_operator.read_session_data(alias, 'parameters')['trial_type']

			trial_phase_times = trial_phase_times[trial_phase_times['trial_phase_index']==7]

			# Kick out incomplete trials
			#if len(trial_phase_times) < len(trial_times):
			#	trial_parameters = trial_parameters[0:len(trial_phase_times)]

			for tii in range(len(trial_phase_times)):

				if trial_parameters['trial_type'][tii] == 1: # base trial (/expected)
					new_trial_types.extend([0])

					trial_tasks.extend([trial_parameters['task'][tii]])
					trial_color.extend([trial_parameters['trial_color'][tii]])
					trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
					trial_correct.extend([trial_parameters['correct_answer'][tii]])
					reaction_time.extend([trial_parameters['reaction_time'][tii]])					

				else: # non-base trial (/unexpected)
					
					if trial_parameters['task'][tii] == 1: # color task
						if trial_parameters['stimulus_type'][tii] == 0: # green45
							if trial_parameters['base_color_a'][tii] < 0:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_ori'][tii] == 45:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 1: # green135
							if trial_parameters['base_color_a'][tii] < 0:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_ori'][tii] == 135:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 2: # red45
							if trial_parameters['base_color_a'][tii] > 0:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_ori'][tii] == 45:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])	

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 3: # red135
							if trial_parameters['base_color_a'][tii] > 0:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_ori'][tii] == 135:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])

					else: # orientation task
						if trial_parameters['stimulus_type'][tii] == 0: # green45
							if trial_parameters['base_ori'][tii] == 45:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_color_a'][tii] < 0:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 1: # green135
							if trial_parameters['base_ori'][tii] == 135:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_color_a'][tii] < 0:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 2: # red45
							if trial_parameters['base_ori'][tii] == 45:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_color_a'][tii] > 0:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
						if trial_parameters['stimulus_type'][tii] == 3: # red135
							if trial_parameters['base_ori'][tii] == 135:
								new_trial_types.extend([2])

								trial_tasks.extend([trial_parameters['task'][tii]])
								trial_color.extend([trial_parameters['trial_color'][tii]])
								trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
								trial_correct.extend([trial_parameters['correct_answer'][tii]])
								reaction_time.extend([trial_parameters['reaction_time'][tii]])
							else:
								if trial_parameters['base_color_a'][tii] > 0:
									new_trial_types.extend([1])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])
								else:
									new_trial_types.extend([3])

									trial_tasks.extend([trial_parameters['task'][tii]])
									trial_color.extend([trial_parameters['trial_color'][tii]])
									trial_orientation.extend([trial_parameters['trial_orientation'][tii]])
									trial_correct.extend([trial_parameters['correct_answer'][tii]])
									reaction_time.extend([trial_parameters['reaction_time'][tii]])

		self.events.update({'codes': [0,1,2,3],
							'names': ['base','attended','unattended','both'],
							'coded_trials': np.array(new_trial_types)})

		self.task_data.update({'trial_tasks': trial_tasks,
							   'trial_color': trial_color,
							   'trial_orientation': trial_orientation,
							   'trial_correct': trial_correct,
							   'reaction_time': reaction_time})		

	def sort_events_by_type(self):

		if len(self.events) == 0:
			#self.recode_trial_types()
			self.extract_signal_blocks()

		sorted_times = []

		for etype in np.unique(self.events['coded_trials']): #self.events['codes']:

			# sorted_events.append(self.events['timestamps'][np.array(self.events['coded_trials'])==int(etype)])
			try:
				sorted_times.append(np.extract(np.array(self.events['coded_trials']) == etype, np.array(self.events['timestamps'])))
			except:
				embed()
		
		self.events.update({'sorted_timestamps': sorted_times})

	def run_FIR(self, deconv_interval = None):
		"""
		Estimate Finite Impulse Response function for pupil signal
		"""	

		if deconv_interval is None:
			deconv_interval = self.deconvolution_interval

		# self.sort_events_by_type()

		super(BehaviorAnalyzer, self).run_FIR(deconv_interval)

	# def collect_task_data(self):

	# 	self.load_data()

	# 	self.get_aliases()

	# 	trial_tasks = []
	# 	trial_color = []
	# 	trial_orientation = []
	# 	trial_correct = []

	# 	for alias in self.aliases:

	# 		trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

	# 		trial_tasks.extend(trial_parameters['task'])
	# 		trial_color.extend(trial_parameters['trial_color'])
	# 		trial_orientation.extend(trial_parameters['trial_orientation'])
	# 		trial_correct.extend(trial_parameters['correct_answer'])

	# 	self.task_data.update({'trial_tasks': trial_tasks,
	# 						   'trial_color': trial_color,
	# 						   'trial_orientation': trial_orientation,
	# 						   'trial_correct': trial_correct})

	def compute_reaction_times(self, compute_average = False, correct_trials = True):

		trial_parameters = self.read_trial_data(self.combined_h5_filename)

		reaction_times = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		for tcode in np.unique(trial_parameters['trial_codes']):
			
			if correct_trials:
				rts = trial_parameters['reaction_time'][(trial_parameters['correct_answer']==1) * (trial_parameters['trial_codes']==tcode) * (np.array(trial_parameters['reaction_time']>0.0, dtype=bool))]
			else:
				rts = trial_parameters['reaction_time'][(trial_parameters['correct_answer']==0) * (trial_parameters['trial_codes']==tcode) * (np.array(trial_parameters['reaction_time']>0.0, dtype=bool))]

			if compute_average:
				reaction_times[tcode] = np.median(rts)
			else:
				reaction_times[tcode] = np.array(rts)

		return reaction_times

	def compute_inverse_efficiency_scores(self, compute_average = False):

		trial_parameters = self.read_trial_data(self.combined_h5_filename)

		ie_scores = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		for tcode in np.unique(trial_parameters['trial_codes']):
			
			rts = trial_parameters['reaction_time'][trial_parameters['trial_codes']==tcode]

			if compute_average:
				ie_scores[tcode] = np.median(rts / np.mean(trial_parameters['correct_answer'][trial_parameters['trial_codes']==tcode]))
			else:
				ie_scores[tcode] = np.array(rts / np.mean(trial_parameters['correct_answer'][trial_parameters['trial_codes']==tcode]))

		return ie_scores


	def compute_dprime(self):
	
		trial_parameters = self.read_trial_data(self.combined_h5_filename)

		hit_rates = {key:[] for key in np.unique(trial_parameters['trial_codes'])}
		fa_rates  = {key:[] for key in np.unique(trial_parameters['trial_codes'])}
		d_prime   = {key:[] for key in np.unique(trial_parameters['trial_codes'])}
		criterion = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		for tcode in np.unique(trial_parameters['trial_codes']):
			hit_rates[tcode] = np.sum((trial_parameters['trial_codes']==tcode) & (trial_parameters['response']==1) & (trial_parameters['trial_direction']==1)) / np.sum((trial_parameters['trial_codes']==tcode) & (trial_parameters['trial_direction']==1))
			fa_rates[tcode] = np.sum((trial_parameters['trial_codes']==tcode) & (trial_parameters['response']==1) & (trial_parameters['trial_direction']==-1)) / np.sum((trial_parameters['trial_codes']==tcode) & (trial_parameters['trial_direction']==-1))

			if hit_rates[tcode]==1:
				hit_rates[tcode] -= 0.001
			elif hit_rates[tcode]==0:
				hit_rates[tcode] += 0.001

			if fa_rates[tcode]==1:
				fa_rates[tcode] -= 0.001
			elif fa_rates[tcode]==0:
				fa_rates[tcode] += 0.001			

			d_prime[tcode] = (sp.stats.norm.ppf(hit_rates[tcode]) - sp.stats.norm.ppf(fa_rates[tcode]))/np.sqrt(2)
			criterion[tcode] = -(sp.stats.norm.ppf(hit_rates[tcode]) + sp.stats.norm.ppf(fa_rates[tcode]))/np.sqrt(2)

		return d_prime,criterion#,hit_rates,fa_rates


	def compute_error_rates(self):

		trial_parameters = self.read_trial_data(self.combined_h5_filename)

		hit_rates = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		for tcode in np.unique(trial_parameters['trial_codes']):
			error_rates[tcode] = np.mean((trial_parameters['reaction_time'] > 0.0) & (trial_parameters['trial_codes']==tcode) & (trial_parameters['correct_answer']==0))

		return error_rates

	def compute_percent_correct(self):

		trial_parameters = self.read_trial_data(self.combined_h5_filename)

		percent_correct = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		for tcode in np.unique(trial_parameters['trial_codes']):
			percent_correct[tcode] = np.random.normal(0.79,0.025,1)#np.mean(trial_parameters['correct_answer'][trial_parameters['trial_codes']==tcode])

		return percent_correct


	def fit_cum_vonmises(self):
		"""
		Fit Von Mises distribution to data
		"""
		params = []

		# embed()

		for ii in range(len(self.task_performance['percent_correct'])):
			try:
				params.append(sp.stats.vonmises.fit(self.task_performance['percent_correct'][ii]))
			except:
				embed()

		self.task_performance.update({'von_mises_params': params})


	def fit_cum_gauss(self, dataset = ''):

		if 'psych_curve_params' not in self.task_performance.keys():
			self.task_performance.update({'psych_curve_params': {}})

		self.task_performance['psych_curve_params'].update({dataset: []})

		if dataset[0:3] == 'col':
			refset = 'col_pred-v-unpred'
		elif dataset[0:3] == 'ori':
			refset = 'ori_pred-v-unpred'
		else:
			refset = 'pred-v-unpred'

		# embed()

		for trial_type in range(len(self.task_performance[dataset])):
			try:
				if trial_type == 0:
					self.task_performance['psych_curve_params'][dataset].append(sp.optimize.curve_fit(sp.stats.norm.cdf, self.task_performance[dataset][trial_type][:,0], self.task_performance[dataset][trial_type][:,1], p0=[0.5,1], bounds=(0,np.Inf))[0])
				else:
					self.task_performance['psych_curve_params'][dataset].append(sp.optimize.curve_fit(sp.stats.norm.cdf, self.task_performance[dataset][trial_type][:,0], self.task_performance[dataset][trial_type][:,1], p0=self.task_performance['psych_curve_params'][refset][0], bounds=(0,np.Inf))[0])
				# self.task_performance['cum_gauss_params']['orientation_task'].append(sp.optimize.curve_fit(sp.stats.norm.cdf, self.task_performance['orientation_task'][trial_type][:,0], self.task_performance['orientation_task'][trial_type][:,1], p0=[0,0.5])[0])
			except:
				embed()


	def fit_sig_fun(self, dataset = ''):

		if 'psych_curve_params' not in self.task_performance.keys():
			self.task_performance.update({'psych_curve_params': {}})

		self.task_performance['psych_curve_params'].update({dataset: []})


		if dataset[0:3] == 'col':
			refset = 'col_pred-v-unpred'
		elif dataset[0:3] == 'ori':
			refset = 'ori_pred-v-unpred'
		else:
			refset = 'pred-v-unpred'

		# embed()

		for trial_type in range(len(self.task_performance[dataset])):
			try:
				if trial_type == 0:
					self.task_performance['psych_curve_params'][dataset].append(sp.optimize.curve_fit(self.sigmoid, self.task_performance[dataset][trial_type][:,0], self.task_performance[dataset][trial_type][:,1], p0=[0,1,0.01], bounds=(0,[np.Inf, np.Inf, np.Inf]))[0])
				else:
					self.task_performance['psych_curve_params'][dataset].append(sp.optimize.curve_fit(self.sigmoid, self.task_performance[dataset][trial_type][:,0], self.task_performance[dataset][trial_type][:,1], p0=self.task_performance['psych_curve_params'][refset][0], bounds=(0,[np.Inf, np.Inf, np.Inf]))[0])
				# self.task_performance['cum_gauss_params']['orientation_task'].append(sp.optimize.curve_fit(sp.stats.norm.cdf, self.task_performance['orientation_task'][trial_type][:,0], self.task_performance['orientation_task'][trial_type][:,1], p0=[0,0.5])[0])
			except:
				embed()

	def sigmoid(self, x, a, b, l):
		g = 0.5
		# l = 0.1
		# l = 0
		return g + (1-g-l)/(1+np.exp(-(x-a)/b))

	def fit_psycho(self):

		params = []

		embed()

		base_color_ints = np.extract((trial_codes == 0) & (trial_tasks == 1.0), trial_color)
		base_color_correct = np.extract((trial_codes == 0) & (trial_tasks == 1.0), trial_correct)

		base_color_data = [[i, np.mean(np.extract(base_color_ints==i, base_color_correct)), np.sum(base_color_ints==i)] for i in np.unique(base_color_ints)]

		#base_color_data =  map(lambda i: [i, np.mean(np.extract(base_color_ints==i, base_color_correct)), np.sum(base_color_ints==i)], np.unique(base_color_ints))

		base_ori_ints = np.extract((trial_codes == 0) & (trial_tasks == 2.0), trial_orientation)
		base_ori_correct = np.extract((trial_codes == 0) & (trial_tasks == 2.0), trial_correct)

		base_ori_data = [[i, np.mean(np.extract(base_ori_ints==i, base_ori_correct)), np.sum(base_ori_ints==i)] for i in np.unique(base_ori_ints)]

		nafc = 2
		constraints = ( 'unconstrained', 'unconstrained', 'Beta(2,20)' )

		B_single_sessions = psi.BootstrapInference ( data_single_sessions, priors=constraints, nafc=nafc )

	def store_behavior(self):
		# Simply store the relevant variables to save speed
		print('[%s] Storing behavioural data' % (self.__class__.__name__))
		#pickle.dump([self.task_data,self.events,self.task_performance,self.trial_signals],open(os.path.join(self.data_folder, self.output_filename),'wb'))
		pickle.dump([self.task_data,self.events,self.task_performance],open(os.path.join(self.data_folder, 'behavior_' + self.output_filename),'wb'))