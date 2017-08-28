

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from numpy import *
import scipy as sp
from pandas import *


from math import *
import os,glob,sys,platform

import pickle as pickle
import pandas as pd

from IPython import embed

# sys.path.append('tools/')
from BehaviorProcessor import BehaviorProcessor
from PupilAnalyzer import PupilAnalyzer
from Plotter import Plotter


from analysis_parameters import *

pl = Plotter(figure_folder = figfolder, linestylemap = linestylemap)

#subname = 'AB'

for subname in sublist:

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	ba = BehaviorProcessor(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

	pa = PupilAnalyzer(subID = subname, filename=h5filename, edf_folder=rawfolder, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

	ba.load_combined_data()
	beh_data = ba.read_trial_data(ba.combined_h5_filename)


	# Get trial information
	trial_codes = ba.recode_trial_code(beh_data)
	beh_data['TR_PE'] = np.array((trial_codes > 1) & (trial_codes >= 30), dtype=int)
	beh_data['TI_PE'] = np.array((trial_codes > 1) & (trial_codes < 30), dtype=int)

	# Some accounting
	beh_data['missed_response'] = np.array(beh_data['button'] == '', dtype=int)
	beh_data['subID'] = sublist.index(subname)


	# Get pupil size change per trial
	pa.compute_TPR(reference_phase=6, time_window = np.array([0.0, 2.0]), sort_by_code = False)

	beh_data['pupil_response'] = 0
	if beh_data.shape[0] > len(pa.TPR):
		beh_data['pupil_response'][0:len(pa.TPR)] = pa.TPR
		beh_data['missed_response'][len(pa.TPR):] = 1
	else:
		beh_data['pupil_response'] = pa.TPR

	# Dump everything to a TSV file
	if sublist.index(subname)==0:
		tsv_output_data = beh_data[['subID','TR_PE','TI_PE','reaction_time','pupil_response','missed_response','correct_answer']]
	else:
		tsv_output_data = tsv_output_data.append(beh_data[['subID','TR_PE','TI_PE','reaction_time','pupil_response','missed_response','correct_answer']], ignore_index = True)


tsv_output_data.to_csv('for_hddm.csv')
