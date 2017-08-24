from __future__ import division

import os,glob,datetime

import numpy as np
import scipy as sp
import seaborn as sn
import matplotlib.pylab as plt
import cPickle as pickle
import pandas as pd
import tables

from scipy.signal import fftconvolve, resample

from math import *

from hedfpy.EDFOperator import EDFOperator
from hedfpy.HDFEyeOperator import HDFEyeOperator
from hedfpy.EyeSignalOperator import EyeSignalOperator

from fir import FIRDeconvolution

from IPython import embed

from Reader import Reader

sn.set(style = 'ticks')

class PupilReader(Reader):

	def __init__(self, subID, filename, edf_folder, sort_by_date = False, verbosity = 0, **kwargs):

		# Setup default parameter values
		self.default_parameters = {'low_pass_pupil_f': 6.0,
								   'high_pass_pupil_f': 0.01}

		super(PupilReader, self).__init__(subID, filename, verbosity=verbosity, **kwargs)

		self.edf_folder = edf_folder
		self.data_folder = edf_folder
		self.sort_by_date = sort_by_date
		self.reference_phase = reference_phase
		self.fir_signal = {}

		# Initialize variables
		self.combined_h5_filename = os.path.join(self.edf_folder,self.subID + '-combined.h5')
		self.combined_data  = None
		self.h5_operator 	= None
		self.pupil_signal 	= None
		self.events 		= {}
		self.pupil_data     = None

	def load_data(self):

		if self.h5_operator is None:
			self.get_h5_operator()

		if not hasattr(self.h5_operator, 'h5f'):
			self.h5_operator.open_hdf_file()

	def unload_data(self):

		if not (self.h5_operator is None):
			self.h5_operator.close_hdf_file()		

		if not (self.combined_data is None):
			self.combined_data.close()

	def load_combined_data(self, force_rebuild = False):

		# if self.combined_data is None:

		if (not os.path.isfile(self.combined_h5_filename)) or force_rebuild: 
			self.recombine_signal_blocks(force_rebuild = force_rebuild)

			#self.combined_data = tables.open_file(self.combined_h5_filename, mode = 'r')


	def read_pupil_data(self, input_file, signal_type = 'full_signal'):

		signal = []
		with tables.open_file(input_file, mode = 'r') as pfile:
			signal = eval('pfile.root.pupil.%s.read()'%signal_type)
			pfile.close()

		return signal

	def read_blink_data(self, input_file, run = 'blinks'):

		with pd.get_store(input_file) as tfile:
			blinks = tfile['pupil/%s/table'%run]

		return blinks

	def read_saccade_data(self, input_file, run = 'saccades'):

		with pd.get_store(input_file) as tfile:
			saccades = tfile['pupil/%s/table'%run]

		return saccades

	def read_trial_data(self, input_file, run = 'full'):

		with pd.get_store(input_file) as tfile:
			params = tfile['trials/' + run + '/table']

		return params

	def get_h5_operator(self):		

		self.h5_operator = HDFEyeOperator(self.data_file)

		if not os.path.isfile(self.data_file):

			edf_files = glob.glob(self.edf_folder + '/*.edf')
			if self.sort_by_date:
				edf_files.sort(key=lambda x: os.path.getmtime(x))
			else:
				edf_files.sort()

			for ii,efile in enumerate(edf_files):

				alias = '%s%d' % (self.subID, ii)

				# insert the edf file contents only when the h5 is not present.
				self.h5_operator.add_edf_file(efile)
				self.h5_operator.edf_message_data_to_hdf(alias = alias)
				self.h5_operator.edf_gaze_data_to_hdf(alias = alias, pupil_hp = self.high_pass_pupil_f, pupil_lp = self.low_pass_pupil_f)

	def get_aliases(self):

		edf_files = glob.glob(self.edf_folder + '/*.edf')
		#edf_files.sort(key=lambda x: os.path.getmtime(x))
		edf_files.sort()

		self.aliases = ['%s%d' % (self.subID, i) for i in range(0, len(edf_files))]			


	def recode_trial_code(self, params, last_node = False):

		#
		# mapping:
		# 0 = expected, color
		# 1 = expected, orientation
		# 
		#     ATT, 		UNATT
		# 10 = pred, 	unpred (color)
		# 30 = unpred, 	pred (color)
		# 50 = unpred, 	unpred (color)
		#
		# 20 = pred, 	unpred (orientation)
		# 40 = unpred, 	pred (orientation)
		# 60 = unpred, 	unpred (orientation)
		#

		# Return array if multiple trials are provided
		if (len(params) > 1) & (~last_node):
			return np.array([self.recode_trial_code(p[1], last_node = True) for p in params.iterrows()], dtype=float)

		new_format = 'trial_cue' in params.keys()

		if not new_format:
			if np.array(params['trial_type'] == 1): # base trial (/expected)
			 	# if np.array(params['task'] == 1):
			 	# 	return 0
			 	# else:
			 	# 	return 1
			 	return np.array(params['task']==2, dtype=int)

			else: # non-base trial (/unexpected)
			 	# if np.array(params['task'] == 1):
			 	# 	return 1
			 	# else:
			 	# 	return 3
				
				if np.array(params['task'] == 1): # color task
					if np.array(params['stimulus_type'] == 0): # red45
						
						if np.array(params['base_color_a'] > 0):
							return 10

						else:
							if np.array(params['base_ori'] == 45):
								return 30
							else:
								return 50
					if np.array(params['stimulus_type'] == 1): # red135
						if np.array(params['base_color_a'] > 0):
							return 10

						else:
							if np.array(params['base_ori'] == 135):
								return 30
							else:
								return 50
					if np.array(params['stimulus_type'] == 2): # green45
						if np.array(params['base_color_a'] < 0):
							return 10

						else:
							if np.array(params['base_ori'] == 45):
								return 30
							else:
								return 50	
					if np.array(params['stimulus_type'] == 3): # green135
						if np.array(params['base_color_a'] < 0):
							return 10

						else:
							if np.array(params['base_ori'] == 135):
								return 30
							else:
								return 50

				else: # orientation task
					if np.array(params['stimulus_type'] == 0): # red45
						if np.array(params['base_ori'] == 45):
							return 20

						else:
							if np.array(params['base_color_a'] > 0):
								return 40
							else:
								return 60
					if np.array(params['stimulus_type'] == 1): # red135
						if np.array(params['base_ori'] == 135):
							return 20

						else:
							if np.array(params['base_color_a'] > 0):
								return 40
							else:
								return 60
					if np.array(params['stimulus_type'] == 2): # green45
						if np.array(params['base_ori'] == 45):
							return 20

						else:
							if np.array(params['base_color_a'] < 0):
								return 40
							else:
								return 60
					if np.array(params['stimulus_type'] == 3): # green135
						if np.array(params['base_ori'] == 135):
							return 20

						else:
							if np.array(params['base_color_a'] < 0):
								return 40
							else:
								return 60				
		else:

			stimulus_types = {'red45': 0,
							  'red135': 1,
							  'green45': 2,
							  'green135': 3}

			if 'trial_cue_label' not in params.keys():
				params['trial_cue'] = stimulus_types[params['trial_cue']]

			if params['trial_cue']==params['trial_stimulus']:
				return np.array(params['task']==2, dtype=int)

			else:
				if np.array(params['task'] == 1): # color task
					if np.array(params['trial_cue'] == 0): # red45
						
						if np.array(params['base_color_a'] > 0):
							return 10

						else:
							if np.array(params['base_ori'] == 45):
								return 30
							else:
								return 50
					if np.array(params['trial_cue'] == 1): # red135
						if np.array(params['base_color_a'] > 0):
							return 10

						else:
							if np.array(params['base_ori'] == 135):
								return 30
							else:
								return 50
					if np.array(params['trial_cue'] == 2): # green45
						if np.array(params['base_color_a'] < 0):
							return 10

						else:
							if np.array(params['base_ori'] == 45):
								return 30
							else:
								return 50	
					if np.array(params['trial_cue'] == 3): # green135
						if np.array(params['base_color_a'] < 0):
							return 10

						else:
							if np.array(params['base_ori'] == 135):
								return 30
							else:
								return 50

				else: # orientation task
					if np.array(params['trial_cue'] == 0): # red45
						if np.array(params['base_ori'] == 45):
							return 20

						else:
							if np.array(params['base_color_a'] > 0):
								return 40
							else:
								return 60
					if np.array(params['trial_cue'] == 1): # red135
						if np.array(params['base_ori'] == 135):
							return 20

						else:
							if np.array(params['base_color_a'] > 0):
								return 40
							else:
								return 60
					if np.array(params['trial_cue'] == 2): # green45
						if np.array(params['base_ori'] == 45):
							return 20

						else:
							if np.array(params['base_color_a'] < 0):
								return 40
							else:
								return 60
					if np.array(params['trial_cue'] == 3): # green135
						if np.array(params['base_ori'] == 135):
							return 20

						else:
							if np.array(params['base_color_a'] < 0):
								return 40
							else:
								return 60


	def recombine_signal_blocks(self, force_rebuild = False):

		if not hasattr(self, 'signal_downsample_factor'):
			self.signal_downsample_factor = 10

		self.get_aliases()

		self.load_data()

		if self.verbosity:
			print '[%s] Recombining signal and parameters from blocks/runs' % (self.__class__.__name__)

		full_signal = np.array([])
		full_baseline = np.array([])
		full_clean = np.array([])

		run_signals = []
		run_baselines = []
		run_saccades = []
		run_cleans = []

		trials = pd.DataFrame()
		blinks = pd.DataFrame()
		saccades = pd.DataFrame()

		run_trials = []
		run_blinks = []

		trial_parameters = {}

		for ii,alias in enumerate(self.aliases):

			this_trial_parameters = self.h5_operator.read_session_data(alias, 'parameters')

			this_trial_phase_times = self.h5_operator.read_session_data(alias, 'trial_phases')
			this_trial_times = self.h5_operator.read_session_data(alias, 'trials')
			
			blocks = self.h5_operator.read_session_data(alias, 'blocks')
			block_start_times = blocks['block_start_timestamp']

			this_block_blinks = self.h5_operator.read_session_data(alias, 'blinks_from_message_file')
			this_block_saccades = self.h5_operator.read_session_data(alias, 'saccades_from_message_file')

			# this_trial_phase_times[this_trial_phase_times['trial_phase_index']==reference_phase]

			# block_event_times = pd.DataFrame()

			this_run_signal = np.array([])
			this_run_clean = np.array([])
			this_run_baseline = np.array([])

			# Kick out incomplete trials
			if len(this_trial_phase_times) < len(this_trial_times):
				this_trial_times = this_trial_times[0:len(this_trial_phase_times)]
				
			this_phase_times = np.array(this_trial_phase_times['trial_phase_EL_timestamp'].values, dtype=float)
			
			prev_signal_size = np.float(full_signal.shape[0])

			for bs,be in zip(blocks['block_start_timestamp'], blocks['block_end_timestamp']):

				this_block_signal = np.squeeze(self.h5_operator.signal_during_period(time_period = (bs, be), alias = alias, signal = 'pupil_bp_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
				this_block_clean_signal = np.squeeze(self.h5_operator.signal_during_period(time_period = (bs, be), alias = alias, signal = 'pupil_bp_clean_zscore', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
				this_block_baseline = np.squeeze(self.h5_operator.signal_during_period(time_period = (bs, be), alias = alias, signal = 'pupil_lp', requested_eye = self.h5_operator.read_session_data(alias,'fixations_from_message_file').eye[0]))
				
				full_signal = np.append(full_signal, this_block_signal)
				full_clean = np.append(full_clean, this_block_clean_signal)
				full_baseline = np.append(full_baseline, this_block_baseline)

				this_run_signal = np.append(this_run_signal, this_block_signal)
				this_run_clean = np.append(this_run_clean, this_block_clean_signal)
				this_run_baseline = np.append(this_run_baseline, this_block_baseline)
				
				this_phase_times[(this_phase_times >= bs) & (this_phase_times < be)] -= bs
				this_block_blinks['start_block_timestamp'] = pd.Series(this_block_blinks['start_timestamp'].values - bs)
				this_block_blinks['end_block_timestamp'] = pd.Series(this_block_blinks['end_timestamp'].values - bs)

				# this_run_blinks = np.append(this_run_blinks, this_)

				this_block_saccades['start_block_timestamp'] = pd.Series(this_block_saccades['start_timestamp'].values - bs)
				this_block_saccades['end_block_timestamp'] = pd.Series(this_block_saccades['end_timestamp'].values - bs)

			this_trial_parameters['trial_codes'] = pd.Series(self.recode_trial_code(this_trial_parameters))
			phases = np.unique(this_trial_phase_times['trial_phase_index'])
			if phases.size>9:
				phases = phases[:9]

			for phase_index in phases:
				this_trial_parameters['trial_phase_%i_within_run'%phase_index] = pd.Series(this_phase_times[np.array(this_trial_phase_times['trial_phase_index']==phase_index, dtype=bool)])
				this_trial_parameters['trial_phase_%i_full_signal'%phase_index] = pd.Series(this_phase_times[np.array(this_trial_phase_times['trial_phase_index']==phase_index, dtype=bool)] + prev_signal_size)	
			
			if ii > 0:

				blinks = pd.concat([blinks, this_block_blinks[this_block_blinks['duration']<4000]], axis=0, ignore_index = True)
				saccades = pd.concat([saccades, this_block_saccades], axis=0, ignore_index = True)

				# Kick out duplicate trials (do we need to do this actually??)
				if this_trial_parameters.iloc[0]['trial_nr'] == trials.iloc[-1]['trial_nr']:
					this_trial_parameters = this_trial_parameters[1:]

				trials = pd.concat([trials, this_trial_parameters], axis=0, ignore_index = True)
			else:
				trials = this_trial_parameters
				blinks = this_block_blinks[this_block_blinks['duration']<4000]
				saccades = this_block_saccades

			run_trials.append(this_trial_parameters)
			run_signals.append(this_run_signal)
			run_cleans.append(this_run_clean)
			run_baselines.append(this_run_baseline)
			run_blinks.append(this_block_blinks[this_block_blinks['duration']<4000])
			run_saccades.append(this_block_saccades)
			
		

		# Store in hdf5 format
		file_mode = 'a'

		if force_rebuild and os.path.isfile(self.combined_h5_filename):
			os.remove(self.combined_h5_filename)

		output_file = tables.open_file(self.combined_h5_filename, mode = file_mode, title = self.subID)

		pgroup = output_file.create_group("/","pupil","pupil")
		tgroup = output_file.create_group("/","trials","trials")

		output_file.create_array(pgroup, "long_signal", full_signal, "long_signal")
		output_file.create_array(pgroup, "clean_signal", full_clean, "clean_signal")
		output_file.create_array(pgroup, "baseline", full_baseline, "baseline")

		for rii in range(len(run_signals)):
			output_file.create_array(pgroup, "r%i_signal"%rii, run_signals[rii], "r%i_signal"%rii)
			output_file.create_array(pgroup, "r%i_clean"%rii, run_signals[rii], "r%i_clean"%rii)
			output_file.create_array(pgroup, "r%i_baseline"%rii, run_baselines[rii], "r%i_baseline"%rii)
		
		output_file.close()
		
		trials.to_hdf(self.combined_h5_filename, key = '/trials/full', mode = 'a', format = 't', data_columns = True)
		blinks.to_hdf(self.combined_h5_filename, key = '/pupil/blinks', mode = 'a', format = 't', data_columns = True)
		saccades.to_hdf(self.combined_h5_filename, key = '/pupil/saccades', mode = 'a', format = 't', data_columns = True)

		for rii in range(len(run_signals)):
			run_trials[rii].to_hdf(self.combined_h5_filename, key = '/trials/run%i'%rii, mode = 'a', format = 't', data_columns = True)
			run_blinks[rii].to_hdf(self.combined_h5_filename, key = '/pupil/r%i_blinks'%rii, mode = 'a', format = 't', data_columns = True)
			run_saccades[rii].to_hdf(self.combined_h5_filename, key = '/pupil/r%i_saccades'%rii, mode = 'a', format = 't', data_columns = True)


	def store_pupil(self):
		# Simply store the relevant variables to save speed
		
		fieldnames = ['task_data','events','trial_signals','fir_signal','pupil_data']

		print '[%s] Storing pupil data' % (self.__class__.__name__)

		output = []

		for fname in fieldnames:
			if hasattr(self, fname):
				print fname
				eval('output.append(self.'+fname+')')

		pickle.dump(output,open(os.path.join(self.data_folder, 'pupil_' + self.output_filename),'wb'))