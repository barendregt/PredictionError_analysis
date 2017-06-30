import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

import statsmodels.api as sm
from statsmodels.stats.api import anova_lm
from statsmodels.formula.api import ols
from math import *

import os,glob,sys

import cPickle as pickle
import pandas as pd

from IPython import embed

alphabetnum = np.array(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), dtype="|S1")

MARKERSIZE = 5

class Plotter(object):

	def __init__(self, figure_folder = '', sn_style='ticks', linestylemap = None):

		sn.set(style = sn_style)

		if len(figure_folder) > 0:
			self.figure_folder = figure_folder
		else:
			self.figure_folder = os.getcwd()

		self.linestylemap = linestylemap

		self.figure = None

	def bootstrap(data, num_samples, statistic, alpha):
	    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
	    n = len(data)
	    idx = np.random.randint(0, n, (num_samples, n))
	    samples = data[idx]
	    stat = np.sort(statistic(samples, 1))
	    return (stat[int((alpha/2.0)*num_samples)],
	            stat[int((1-alpha/2.0)*num_samples)])

	def incorrect_data_format(self, data, conditions):
		if (not isinstance(data, dict)) and (not isinstance(data, pd.DataFrame)):
			print('ERROR: data must be a dictionary or dataframe')
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

	def tsplot(self, data, tnames = [], time = [], name = 'Time (au)', ci=[95], legend=True):
		sn.tsplot(data = data, condition = tnames, time = time, name= name, ci=ci, legend=legend)
		sn.despine(offset=5)

	def event_related_pupil_average(self, data, conditions = [], signal_labels = [], xtimes = [], yticks = [], xticks = [], x_lim =[None, None], y_lim=[None, None], yticklabels = [], xticklabels = [], onset_marker = [], xlabel = 'Time (s)', ylabel = 'Pupil size (sd)', show_legend = False, title = '', compute_mean = False, compute_sd = False):
			

		if onset_marker != []:
			plt.axvline(onset_marker, ymin=0, ymax=1, linewidth=1.5, color='k', figure = self.figure)

		if isinstance(data, dict):
			for (label, signal) in data.items():
				if label in conditions:
					if compute_mean:
						msignal = np.mean(signal, axis=0)
					else:
						msignal = signal

					if compute_sd:
						ste_signal = np.array(signal)
						#condition_ste = np.std(signal, axis=0)/np.sqrt(len(signal))			
						condition_ste = np.zeros((2,signal.shape[1]))			
						for t in range(signal.shape[1]):
							condition_ste[:,t] = self.bootstrap(signal[:,t], 1000, mean, 0.05)

						if self.linestylemap is None:
							plt.fill_between(range(len(msignal)), condition_ste[0], condition_ste[1], alpha=0.1)
						else:
							plt.fill_between(range(len(msignal)), condition_ste[0], condition_ste[1], alpha=0.1, color=self.linestylemap[label][0])		

					if not signal_labels:
						if self.linestylemap is None:
							self.plot(xtimes, msignal, label=label)
						else:
							self.plot(xtimes, msignal, label=label, color=self.linestylemap[label][0], ls=self.linestylemap[label][1], marker=self.linestylemap[label][2], markersize=MARKERSIZE, markeredgecolor=self.linestylemap[label][3], markerfacecolor=self.linestylemap[label][4])
					else:
						if self.linestylemap is None:
							self.plot(xtimes, msignal, label=signal_labels[label])
						else:
							self.plot(xtimes, msignal, label=signal_labels[label], color=self.linestylemap[label][0], ls=self.linestylemap[label][1], marker=self.linestylemap[label][2], markersize=MARKERSIZE, markeredgecolor=self.linestylemap[label][3], markerfacecolor=self.linestylemap[label][4])

				

		else:
			for (key,signal) in enumerate(data):
				if key in conditions:
					if compute_mean:
						msignal = np.mean(signal, axis=0)
					else:
						msignal = signal

					if compute_sd:	
						condition_ste = np.std(signal, axis=0)/np.sqrt(len(signal))
						if self.linestylemap is None:
							plt.fill_between(range(len(msignal)), msignal-condition_ste, msignal+condition_ste, alpha=0.1)	
						else:
							plt.fill_between(range(len(msignal)), msignal-condition_ste, msignal+condition_ste, alpha=0.1, color=self.linestylemap[key][0])	

					if not signal_labels:
						if self.linestylemap is None:
							self.plot(xtimes, msignal, label=label)
						else:
							self.plot(xtimes, msignal, label=label, color=self.linestylemap[key][0], ls=self.linestylemap[key][1], marker=self.linestylemap[key][2], markersize=MARKERSIZE, markeredgecolor=self.linestylemap[key][3], markerfacecolor=self.linestylemap[key][4])
					else:
						if self.linestylemap is None:
							self.plot(xtimes, msignal, label=signal_labels[key])
						else:
							self.plot(xtimes, msignal, label=signal_labels[key], color=self.linestylemap[key][0], ls=self.linestylemap[key][1], marker=self.linestylemap[key][2], markersize=MARKERSIZE, markerfacecolor=self.linestylemap[key][4], markeredgecolor=self.linestylemap[key][3])

	
		
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

		plt.xlim(x_lim[0], x_lim[1])
		plt.ylim(y_lim[0], y_lim[1])	

		sn.despine(offset=5)		


	def event_related_pupil_difference(self, data, conditions, reference_index = 0, xtimes = [], x_lim =[None, None], y_lim=[None, None], yticks = [], xticks = [], yticklabels = [], xticklabels = [], show_legend = True, title='', ylabel = '', xlabel = '', with_stats = False, with_error = False):
		
		# Plot the difference between conditions

		if self.incorrect_data_format(data, conditions):
			return

		# default behaviour is: diff_N = condition_N - condition_0

		reference_condition = conditions[reference_index]
		#sub_ii_list = np.arange(0,len(data[reference_condition]), 2)
		reference_mean = np.mean(data[reference_condition], axis=0)
		
		for key in conditions:
			if key != reference_condition:
				condition_mean = np.mean(reference_mean - data[key],axis=0) 
				self.plot(xtimes, condition_mean, label=reference_condition+'v'+key, color=self.linestylemap[key][0], ls=self.linestylemap[key][1], marker=self.linestylemap[key][2], markersize=MARKERSIZE, markeredgecolor=self.linestylemap[key][3], markerfacecolor=self.linestylemap[key][4])

				if with_error:
					condition_ste = np.std(reference_mean - data[key], axis=0)/np.sqrt(len(data[key]))
					plt.fill_between(range(np.array(reference_mean).size), condition_mean-condition_ste, condition_mean+condition_ste, alpha=0.5, color=self.linestylemap[key][0])

		
		# Do time-by-time stats on difference
		
		if with_stats:
			extract_data = np.array([data[key] for key in conditions])

			f = np.zeros((np.array(reference_mean).size,1))
			p = np.zeros((np.array(reference_mean).size,1))

			y_pos = plt.axis()[2]

			for time_point in range(np.array(reference_mean).size):
				y_pos = 0
				# All conditions one-way
				f[time_point],p[time_point] = sp.stats.f_oneway(extract_data[0][time_point],
																  extract_data[1][time_point],
																  extract_data[2][time_point],
																  extract_data[3][time_point])

				if p[time_point] < (0.000000000001):
					plt.text(time_point, y_pos,'*')																	


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

		plt.xlim(x_lim[0], x_lim[1])
		plt.ylim(y_lim[0], y_lim[1])	

		# plt.axis('square')

		sn.despine(offset=5)


	def bar_plot(self, data, conditions, with_error = False, with_stats = False, with_data_points = False, ylabel = '', xlabel = '', yticks = [], xticks = [], yticklabels = [], xticklabels = [], x_lim = [None, None], y_lim = [None, None]):
		if self.incorrect_data_format(data, conditions):
			return

		latex_code = '\\addplot plot coordinates {'

		if self.linestylemap is None:
			bar_color = {key:'w' for key in conditions}
		else:
			bar_color = {key: self.linestylemap[key][0] for key in conditions}

		bar_width = 0.75

		for ii,key in enumerate(conditions):

			if with_error:
				plt.bar(ii+1, np.nanmean(data[key]), yerr = np.nanstd(data[key])/np.sqrt(len(data[key])), width = bar_width, label = key, color= bar_color[key],edgecolor='k')
			else:
				plt.bar(ii+1, np.nanmean(data[key]), width = bar_width, label = key, color = bar_color[key], edgecolor='k')
			
			if with_data_points:
				plt.plot(np.ones((len(data[key]),1))*ii+1, data[key], 'o')

			latex_code += '(%i,%.2f) '%(ii, np.mean(data[key]))

		# plt.xticks(1+np.arange(len(conditions)), conditions)
		latex_code += '};'

		print latex_code

		plt.ylabel(ylabel)
		plt.xlabel(xlabel)

		if len(xticks)>0:
			if len(xticklabels)>0:
				plt.xticks(xticks + (bar_width/2), xticklabels)
			else:
				plt.xticks(xticks + (bar_width/2))
		else:
			if len(xticklabels)>0:
				plt.xticks(1+np.arange(len(conditions)) + (bar_width/2), xticklabels)
			else:
				plt.xticks(1+np.arange(len(conditions)) + (bar_width/2), conditions)

		if len(yticks)>0:
			if len(yticklabels)>0:
				plt.yticks(yticks, yticklabels)
			else:
				plt.yticks(yticks)			

		plt.xlim(x_lim[0], x_lim[1])
		plt.ylim(y_lim[0], y_lim[1])

		# plt.axis('square')

		sn.despine()

	def factor_plot(data, *args, **kwargs):
		# Draw a nested barplot to show survival for class and sex
		g = sn.factorplot(data=data, *args, **kwargs)
		g.despine(left=True)

	def hline(self, y = 0, label = None):
		
		if label is not None:
			plt.text(0.55, y, label, alpha = 0.5, fontsize=8, horizontalalignment='left', verticalalignment='center', bbox=dict(facecolor='w',edgecolor='w'))

		plt.axhline(y = y, color='k', linewidth = 0.75, figure=self.figure, linestyle='dashed', alpha=0.5)	

	def vline(self, x = 0, label = None):
		plt.axvline(x = x, color='k', linewidth = 0.75, figure=self.figure, linestyle='dashed', alpha=0.5)

		if label is not None:
			plt.text(x, -0.1, label)

	def save_figure(self, filename = '', sub_folder = ''):

		self.figure.tight_layout()

		# Create a random PDF filename if none is provided
		if len(filename)==0: 
			filename = "".join(np.random.choice(alphabetnum, [1, 8])) + '.pdf'

		if len(sub_folder) > 0:
			if not os.path.isdir(os.path.join(self.figure_folder, sub_folder+'/')):
				os.makedirs(os.path.join(self.figure_folder, sub_folder+'/'))

			self.figure.savefig(os.path.join(self.figure_folder, sub_folder, filename))
		else:
			self.figure.savefig(os.path.join(self.figure_folder, filename))