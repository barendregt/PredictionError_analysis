# some parameters used by multiple analysis scripts

import numpy as np
import platform

if platform.node()=="aeneas":
	raw_data_folder = '/home/raw_data/2017/visual/PredictionError/Behavioural/Reaction_times/combined/'
else:
	raw_data_folder = '/home/barendregt/Projects/PredictionError/Psychophysics/Data/combined/' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/PredictionError/Figures'



low_pass_pupil_f, high_pass_pupil_f = 5.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
response_deconvolution_interval = np.array([-1.5, 4.5])
stimulus_deconvolution_interval = np.array([-1, 4.5])

down_fs = int(signal_sample_frequency / deconv_sample_frequency)


sublist = ['AA','AB','AC','AF','AG','AH','AI','AJ','AK','AL','AM','AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ','BA','IAA','IAB','IAC','IAF','IAG','IAH','IAI','IAJ','IAK','IAL','IAM','IAN','IAO','IAP','IAQ','IAR','IAS']