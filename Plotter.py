import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

from math import *
import os,glob,sys

import cPickle as pickle
import pandas as pd

from IPython import embed


class Plotter(object):

	def __init__(self, figure_folder = '', sn_style='ticks'):

		sn.set(style = sn_style)

		if len(figure_folder) > 0:
			self.figure_folder = figure_folder
		else:
			self.figure_folder = os.getcwd()


	def event_related_pupil_average(self, pupil_signal = [], xtimes = []):
		
		if len(xtimes)==0:
			xtimes = np.arange(pupil_signal.shape[1])

		pred_signal = []
		unpred_signal = []

		for key,trial_signal in pa.trial_signals.items():
			if key < 10:
				pred_signal.extend(trial_signal - trial_signal[:,:5].mean())
			else:
				unpred_signal.extend(trial_signal - trial_signal[:,:5].mean())

		plt.axvline(5, ymin=0, ymax=1, linewidth=1.5, color='k')

		plt.plot(np.mean(pred_signal, axis=0), label='expected')
		plt.plot(np.mean(unpred_signal, axis=0), label='unexpected')

		plt.xticks()

		plt.ylabel('Pupil response (z)')
		plt.xlabel('Time after stim onset (s)')

		plt.xticks(np.arange(5,45,10).tolist(),[0, 1.0, 2.0, 3.0, 4.0])#, labels=np.arange(-0.5,4.0,5).tolist())

		# plt.legend()

		sn.despine(offset=5)		
