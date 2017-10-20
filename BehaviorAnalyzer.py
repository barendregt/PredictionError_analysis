# from __future__ import division
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
from BehaviorProcessor import BehaviorProcessor

class BehaviorAnalyzer(PupilAnalyzer):

	def __init__(self, subID, csv_filename, h5_filename, raw_folder, verbosity=0, **kwargs):

		self.default_parameters = {}

		super(BehaviorAnalyzer, self).__init__(subID, h5_filename, raw_folder,verbosity=verbosity, **kwargs)


		self.BP = BehaviorProcessor(subID, csv_filename, h5_filename, raw_folder, verbosity=verbosity, **kwargs)

		self.csv_file = csv_filename

		self.h5_operator = None
		self.task_data = {}

		self.task_performance = {}


	def compute_reaction_times(self, compute_average = False, correct_trials = True):

		trial_parameters = self.PR.read_trial_data(self.PR.combined_h5_filename)

		valid_trials = trial_parameters['reaction_time'] > 0.0

		reaction_times = pd.DataFrame(data = np.vstack([trial_parameters['reaction_time'][valid_trials].values, trial_parameters['trial_codes'][valid_trials].values, trial_parameters['correct_answer'][valid_trials].values]).T, columns = ['reaction_time','trial_code','correct'])

		return reaction_times

	def compute_inverse_efficiency_scores(self, compute_average = False):

		trial_parameters = self.PR.read_trial_data(self.PR.combined_h5_filename)

		ie_scores = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		for tcode in np.unique(trial_parameters['trial_codes']):
			
			rts = trial_parameters['reaction_time'][trial_parameters['trial_codes']==tcode]

			if compute_average:
				ie_scores[tcode] = np.median(rts / np.mean(trial_parameters['correct_answer'][trial_parameters['trial_codes']==tcode]))
			else:
				ie_scores[tcode] = np.array(rts / np.mean(trial_parameters['correct_answer'][trial_parameters['trial_codes']==tcode]))

		return ie_scores


	def compute_dprime(self):
	
		trial_parameters = self.PR.read_trial_data(self.PR.combined_h5_filename)

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

		trial_parameters = self.PR.read_trial_data(self.PR.combined_h5_filename)

		hit_rates = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		for tcode in np.unique(trial_parameters['trial_codes']):
			error_rates[tcode] = np.mean((trial_parameters['reaction_time'] > 0.0) & (trial_parameters['trial_codes']==tcode) & (trial_parameters['correct_answer']==0))

		return error_rates

	def compute_percent_correct(self):

		trial_parameters = self.PR.read_trial_data(self.PR.combined_h5_filename)

		percent_correct = {key:[] for key in np.unique(trial_parameters['trial_codes'])}

		for tcode in np.unique(trial_parameters['trial_codes']):
			percent_correct[tcode] = np.mean(trial_parameters['correct_answer'][trial_parameters['trial_codes']==tcode]) #np.random.normal(0.79,0.025,1)#

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

		if 'psych_curve_params' not in list(self.task_performance.keys()):
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

		if 'psych_curve_params' not in list(self.task_performance.keys()):
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
		print(('[%s] Storing behavioural data' % (self.__class__.__name__)))
		#pickle.dump([self.task_data,self.events,self.task_performance,self.trial_signals],open(os.path.join(self.data_folder, self.output_filename),'wb'))
		pickle.dump([self.task_data,self.events,self.task_performance],open(os.path.join(self.data_folder, 'behavior_' + self.output_filename),'wb'))