import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

from math import *
import os,glob,sys

import pickle as pickle
import pandas as pd

from IPython import embed

# sys.path.append('tools/')
from BehaviorAnalyzer import BehaviorAnalyzer


raw_data_folder = '/home/barendregt/Projects/Attention_Prediction/Psychophysics/Data' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/PredictionError/Figures'

sublist = ['AA','AB','AC','AF','AG']#
# sublist = ['s1','s2','s3','s4','s5','s6']#['s1','s2','s3','s4','s5','s6']#['s1','s2','s4']['s1','s2',[

# subname = 's1'#'tk2'#'s3'#'mb2'#'tk2'#'s3'#'tk2'

low_pass_pupil_f, high_pass_pupil_f = 6.0, 0.01

signal_sample_frequency = 1000
deconv_sample_frequency = 10
deconvolution_interval = np.array([0.0, 6.0])

down_fs = 100

sn.set(style='ticks')

all_sub_rts = [[],[],[]]
 
def run_analysis(subname):	

	print(('[main] Running analysis for %s' % (subname)))

	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)

	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]

	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 3, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = deconvolution_interval)

	# pa.recombine_signal_blocks()
	# Pupil data
	pa.load_data()

	pa.signal_per_trial()

	pa.get_IRF()

	# plt.figure(figsize=(10,8))
	plt.subplot(2,3,sublist.index(subname)+1)

	sub_HRF = {'stimulus': [], 'button_press': []}

	for name,dec in zip(list(pa.FIRo.covariates.keys()), pa.FIRo.betas_per_event_type.squeeze()):
		#pa.fir_signal.update({name: [pa.FIRo.deconvolution_interval_timepoints, dec]})
		plt.plot(pa.FIRo.deconvolution_interval_timepoints, dec, label = name)

		sub_HRF[name] = dec


	plt.legend()	
	sn.despine(offset=5)

	pa.build_design_matrix(sub_HRF)
	pa.run_GLM()


plt.figure()
[run_analysis(sub) for sub in sublist]
plt.tight_layout()

plt.savefig(os.path.join(figfolder,'all_sub_IRF.pdf'))