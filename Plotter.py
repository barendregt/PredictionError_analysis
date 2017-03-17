import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

from math import *
import os,glob,sys

import cPickle as pickle
import pandas as pd

from IPython import embed

alphabetnum = np.array(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), dtype="|S1")

class Plotter(object):

	def __init__(self, figure_folder = '', sn_style='ticks'):

		sn.set(style = sn_style)

		if len(figure_folder) > 0:
			self.figure_folder = figure_folder
		else:
			self.figure_folder = os.getcwd()


		self.figure = None

	def incorrect_data_format(self, data, conditions):
		if not isinstance(data, dict):
			print('ERROR: data must be a dictionary')
			return True

		# Quickly check that all requested conditions are present in the data
		for key in conditions:
			if key not in data.keys():
				print('ERROR: not all requested keys present in provided data')
				return True
		return False

	def open_figure(self, force = 0):

		if self.figure is None:
			self.figure = plt.figure()
		else:
			if force==1:
				plt.close("all")

				self.figure = plt.figure()
			else:
				print('Warning: figure is already open. Use force=1 to create new axes')

	def subplot(self, *args, **kwargs):
		plt.subplot(*args, **kwargs)		

	def plot(self, x, y, label = None, *args, **kwargs):

		if len(x)==0:
			plt.plot(np.arange(np.array(y).size), y, label=label, figure = self.figure, *args, **kwargs)
		else:
			plt.plot(x, y, label=label, figure = self.figure, *args, **kwargs)

	def event_related_pupil_average(self, pupil_signal, signal_labels = {}, xtimes = [], yticks = [], xticks = [], yticklabels = [], xticklabels = [], onset_marker = [], xlabel = 'Time (s)', ylabel = 'Pupil size (sd)', show_legend = True, title = '', compute_mean = False, compute_sd = False):
			

		if onset_marker != []:
			plt.axvline(onset_marker, ymin=0, ymax=1, linewidth=1.5, color='k', figure = self.figure)

		if isinstance(pupil_signal, dict):
			for (label, signal) in pupil_signal.items():

				if compute_mean:
					signal = np.mean(signal, axis=0)

					self.plot(xtimes, signal, label=label)

		else:
			for (key,signal) in enumerate(pupil_signal):
				if compute_mean:
					signal = np.mean(signal, axis=0)

				if len(signal_labels)==0:
					self.plot(xtimes, signal, label=key)
				else:
					self.plot(xtimes, signal, label=signal_labels[key])
		
		plt.ylabel(ylabel)
		plt.xlabel(xlabel)

		if len(title)>0:
			plt.title(title)

		if show_legend:
			plt.legend(loc = 'best')

		if len(xticks)>0:
			if len(xticklabels)>0:
				plt.xticks(xticks, xticklabels)
			else:
				plt.xticks(xticks)

		if len(yticks)>0:
			if len(yticklabels)>0:
				plt.yticks(yticks, yticklabels)
			else:
				plt.yticks(yticks)

		sn.despine(offset=5)		


	def event_related_pupil_difference(self, data, conditions, reference_index = 0, xtimes = [], yticks = [], xticks = [], yticklabels = [], xticklabels = [], show_legend = True, title='', ylabel = '', xlabel = '', with_stats = False):
		
		# Plot the difference between conditions

		if self.incorrect_data_format(data, conditions):
			return
		
		# default behaviour is: diff_N = condition_N - condition_0
		reference_condition = conditions[reference_index]
		reference_mean = np.mean(data[reference_condition], axis=0)
		
		for key in conditions[1:]:
			condition_mean = np.mean(np.array(data[key]) - reference_mean, axis=0) 
			self.plot(xtimes, condition_mean, label=reference_condition+'v'+key)

			# if with_stats:
			# 	condition_ste = np.std(np.array(data[key]) - reference_mean, axis=0)/np.sqrt(len(data[key]))
			# 	plt.fill_between(range(np.array(reference_mean).size), condition_mean-condition_ste, condition_mean+condition_ste, alpha=0.5)

		
		# Do time-by-time stats on difference
		
		if with_stats:
			extract_data = np.array([data[key] for key in conditions])

			f = np.zeros((np.array(reference_mean).size,1))
			p = np.zeros((np.array(reference_mean).size,1))

			y_pos = plt.axis()[2]

			for time_point in range(np.array(reference_mean).size):
				y_pos = 0
				# All conditions one-way
				f[time_point],p[time_point] = sp.stats.f_oneway(extract_data[0][:][time_point],
																  extract_data[1][:][time_point],
																  extract_data[2][:][time_point],
																  extract_data[3][:][time_point])

				if p[time_point] < (0.000000000001):
					plt.text(time_point, y_pos,'*')

				# y_pos = 1
				# for ii in range(4):
				# 	for jj in range(4):
				# 		if (ii<jj):
				# 			# All combinations
				# 			f[time_point],p[time_point] = sp.stats.f_oneway(extract_data[ii][:][time_point],
				# 															  extract_data[jj][:][time_point])

				# 			if p[time_point] < (0.05/8):
				# 				plt.text(time_point, y_pos,'*')	

				# 			y_pos += 1

				# plt.axis([0, len(reference_mean), 0, y_pos+1])

			print(p)																		


		if len(title)>0:
			plt.title(title)

		if show_legend:
			plt.legend(loc = 'best')

		plt.ylabel(ylabel)
		plt.xlabel(xlabel)

		if len(xticks)>0:
			if len(xticklabels)>0:
				plt.xticks(xticks, xticklabels)
			else:
				plt.xticks(xticks)

		if len(yticks)>0:
			if len(yticklabels)>0:
				plt.yticks(yticks, yticklabels)
			else:
				plt.yticks(yticks)		

		sn.despine(offset=5)

	def pupil_amplitude_per_condition(self, data, conditions, with_error = False):

		if self.incorrect_data_format(data, conditions):
			return

		for ii,key in enumerate(conditions):

			# First compute the within-subject average
			# average_data = [np.mean(subdata) for subdata in data[key]]

			# Then plot group average
			if with_error:
				plt.bar(ii, np.mean(data[key]), yerr = np.std(data[key])/np.sqrt(len(data[key])), width = 0.75, label = key)
			else:
				plt.bar(ii, np.mean(data[key]), width = 0.75, label = key)

		plt.xticks(range(len(conditions)), conditions)

		sn.despine()


	def save_figure(self, filename = ''):

		self.figure.tight_layout()

		# Create a random PDF filename if none is provided
		if len(filename)==0: 
			filename = "".join(np.random.choice(alphabetnum, [1, 8])) + '.pdf'

		self.figure.savefig(os.path.join(self.figure_folder, filename))