# some parameters used by multiple analysis scripts

import numpy as np
import platform

if platform.node()=="aeneas":
	raw_data_folder = '/home/raw_data/2017/visual/PredictionError/Behavioural/Reaction_times/combined/'
else:
	raw_data_folder = '/home/barendregt/Projects/PredictionError/Psychophysics/Data/combined/' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/PredictionError/Figures'



low_pass_pupil_f, high_pass_pupil_f = 4.0, 0.05

signal_sample_frequency = 1000
deconv_sample_frequency = 10
response_deconvolution_interval = np.array([-1.5, 4.5])
stimulus_deconvolution_interval = np.array([-1, 4.5])
trial_deconvolution_interval = np.array([-0.5, 5.0])

down_fs = int(signal_sample_frequency / deconv_sample_frequency)

linestylemap = {'PP': [[.5,.5,.5],'-',None,None,None],
				'UP': [[.3,.3,.3],'-','o','w','k'],
				'PU': [[.3,.3,.3],'-','o','k','w'],
				'UU': ['c','--','o','c','c']}


sublist = ['AB','AC','AF','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AS','AT','AV','AX','AZ']#,'BA','IAA','IAC','IAF','IAH','IAJ','IAK','IAL','IAM','IAN','IAO','IAP','IAQ']