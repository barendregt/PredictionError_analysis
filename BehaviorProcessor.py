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

from PupilProcessor import PupilProcessor

class BehaviorProcessor(PupilProcessor):

	def __init__(self, subID, csv_filename, h5_filename, raw_folder, verbosity=0, **kwargs):

		self.default_parameters = {}

		super(BehaviorProcessor, self).__init__(subID, h5_filename, raw_folder,verbosity=verbosity, **kwargs)

		self.csv_file = csv_filename

		self.h5_operator = None
		self.task_data = {}

		self.task_performance = {}

	def load_data(self):

		super(BehaviorProcessor, self).load_data()

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


		if ('trial_cue' in list(self.csv_data.keys())) and ('trial_cue_label' not in list(self.csv_data.keys())):
			stimulus_types = {'red45': 0,
							  'red135': 1,
							  'green45': 2,
							  'green135': 3}

			self.csv_data['trial_cue_label'] = self.csv_data['trial_cue']
			self.csv_data['trial_cue'] = np.array([stimulus_types[tc] for tc in self.csv_data['trial_cue_label']], dtype=int)

	def to_tsv(self):
		"""
		Output the data in TSV format (with or without additional processing)
		"""

		

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


	def store_behavior(self):
		# Simply store the relevant variables to save speed
		print(('[%s] Storing behavioural data' % (self.__class__.__name__)))
		#pickle.dump([self.task_data,self.events,self.task_performance,self.trial_signals],open(os.path.join(self.data_folder, self.output_filename),'wb'))
		pickle.dump([self.task_data,self.events,self.task_performance],open(os.path.join(self.data_folder, 'behavior_' + self.output_filename),'wb'))