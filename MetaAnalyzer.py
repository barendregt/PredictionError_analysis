
from PupilAnalyzer import PupilAnalyzer

import * from analysis_parameters

class MetaAnalyzer(object):

	def __init__(self, sublist, **kwargs):
		self.sublist = sublist


	def average_pupil_response(self):
		pupil_signals = {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)}		 


		#		TASK:  		  COLOR	    ORI
		condition_keymap = { 0: 'PP',  1: 'PP',
							10: 'PU', 20: 'PU',
							30: 'UP', 40: 'UP',
							50: 'UU', 60: 'UU'}

		inverse_keymap = {'PP': [0,1],
						  'UP': [30,40],
						  'PU': [10,20],
						  'UU': [50,60]}

		for subname in self.sublist:

			# Organize filenames
			rawfolder = os.path.join(raw_data_folder,subname)
			sharedfolder = os.path.join(shared_data_folder,subname)
			csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
			h5filename = os.path.join(sharedfolder,subname+'.h5')

			# Initialize PA object
			pa = PupilAnalyzer(subname, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = stimulus_deconvolution_interval, verbosity = 0)


			# Combine signals based on condition	

			sub_signals = {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
										 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
										 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
										 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)}		 

			# Get trial-based, event-related, baseline-corrected signals centered on stimulus onset
			pa.signal_per_trial(only_correct = False, only_incorrect = False, reference_phase = 7, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = False, return_rt = True)

			for (key,signals) in pa.trial_signals.items():
				if len(signals)>0:
					sub_signals[condition_keymap[key]] = np.append(sub_signals[condition_keymap[key]], signals, axis=0)					

			for con in inverse_keymap.keys():
				pupil_signals[con] = np.append(pupil_signals[con], sub_signals[con].mean(axis=0)[np.newaxis,:], axis=0)				

		return pupil_signals

	def average_pupil_response_by_correctness(self):
		pupil_signals = {'correct': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)},
				 'incorrect': {'PP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UP': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'PU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float),
				 'UU': np.empty((0,int((stimulus_deconvolution_interval[1] - stimulus_deconvolution_interval[0])*signal_sample_frequency)),dtype=float)}}		 


		#		TASK:  		  COLOR	    ORI
		condition_keymap = { 0: 'PP',  1: 'PP',
							10: 'PU', 20: 'PU',
							30: 'UP', 40: 'UP',
							50: 'UU', 60: 'UU'}

		inverse_keymap = {'PP': [0,1],
						  'UP': [30,40],
						  'PU': [10,20],
						  'UU': [50,60]}

		for subname in self.sublist:

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

			# Get trial-based, event-related, baseline-corrected signals centered on stimulus onset
			pa.signal_per_trial(only_correct = True, only_incorrect = False, reference_phase = 7, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = False, return_rt = True)

			for (key,signals) in pa.trial_signals.items():
				if len(signals)>0:
					sub_signals['correct'][condition_keymap[key]] = np.append(sub_signals['correct'][condition_keymap[key]], signals, axis=0)					

			# Get trial-based, event-related, baseline-corrected signals centered on stimulus onset
			pa.signal_per_trial(only_correct = False, only_incorrect = True, reference_phase = 7, with_rt = False, baseline_type = 'relative', baseline_period = [-.5, 0.0], force_rebuild=False, down_sample = False, return_rt = True)

			for (key,signals) in pa.trial_signals.items():
				if len(signals)>0:
					sub_signals['incorrect'][condition_keymap[key]] = np.append(sub_signals['incorrect'][condition_keymap[key]], signals, axis=0)

			for con in inverse_keymap.keys():
				pupil_signals['correct'][con] = np.append(pupil_signals['correct'][con], sub_signals['correct'][con].mean(axis=0)[np.newaxis,:], axis=0)
				pupil_signals['incorrect'][con] = np.append(pupil_signals['incorrect'][con], sub_signals['incorrect'][con].mean(axis=0)[np.newaxis,:], axis=0)	

		return pupil_signals

	def trial_pupil_response_by_correctness(self):

		
		

		return pupil_signals		