import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sn

import statsmodels.api as sm
from statsmodels.stats.api import anova_lm
from statsmodels.formula.api import ols
from math import *

import os,glob,sys

import pickle as pickle
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

	def bootstrap(self, data, num_samples, statistic, alpha):
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
			if key not in list(data.keys()):
				print('ERROR: not all requested keys present in provided data')
				return True
		return False

	def open_figure(self, force = 0, visible = False):

		if self.figure is None:
			self.figure = plt.figure()
		else:
			if force==1:
				plt.close("all")

				self.figure = plt.figure()
			else:
				print('Warning: figure is already open. Use force=1 to create new axes')

		if visible:
			plt.show(block=False)

	def subplot(self, *args, **kwargs):
		return plt.subplot(*args, **kwargs)		

	def plot(self, x, y, label = None, *args, **kwargs):

		if len(x)==0:
			plt.plot(np.arange(np.array(y).size), y, label=label, figure = self.figure, *args, **kwargs)
		else:
			plt.plot(x, y, label=label, figure = self.figure, *args, **kwargs)

	def tsplot(self, data, tnames = [], time = [], name = 'Time (au)', ci=[95], legend=True):
		sn.tsplot(data = data, condition = tnames, time = time, name= name, ci=ci, legend=legend)
		sn.despine(offset=5)
		
	def event_related_pupil_average(self, data, conditions = [], signal_labels = [], 
									xtimes = [], yticks = [], xticks = [], x_lim =[None, None], y_lim=[None, None], axis_ratio = None,
									yticklabels = [], xticklabels = [], onset_marker = [], xlabel = 'Time (s)', ylabel = 'Pupil size (sd)', 
									title = '', show_legend = False, legend_prop = None, legend_loc = 'best', legend_fontsize = None,
									compute_mean = False, compute_sd = False, bootstrap_sd = False, with_stats = False, stats_ttest_ref = 0.0, stats_ttest_p = 0.05, sig_marker_ypos = 0.0, jitter_sig_markers = False, mark_first_sig = False, report_pvals = False,
									smooth_signal = False, smooth_factor = 10, after_smooth = False, after_smooth_window = 11,
									dt = False):
			

		if onset_marker != []:
			plt.axvline(onset_marker, ymin=0, ymax=1, linewidth=1.5, color='k', figure = self.figure)

		# Window size must be odd
		if (after_smooth_window%2)==0:
			after_smooth_window += 1

		if isinstance(data, dict):
			for (label, signal) in list(data.items()):
				if (len(conditions)==0) | (label in conditions):

					if smooth_signal:
						signal = sp.signal.decimate(signal, smooth_factor, 1)

					if compute_sd:
						# if smooth_signal:
						# 	ste_signal = sp.signal.decimate(np.array(signal),smooth_factor,axis=1)
						# else:

						if after_smooth:
							ste_signal = sp.signal.savgol_filter(signal, after_smooth_window, 2)
						else:
							ste_signal = signal#np.array(signal)
						if bootstrap_sd:
									
							condition_ste = np.zeros((2,ste_signal.shape[1]))			
							for t in range(ste_signal.shape[1]):
								condition_ste[:,t] = self.bootstrap(ste_signal[:,t], 1000, np.nanmean, 0.05)
						else:
							tmp_ste = ste_signal.std(axis=0)/np.sqrt(ste_signal.shape[0])#np.std(signal, axis=0)/np.sqrt(len(signal))
							condition_ste = np.array([np.nanmean(ste_signal, axis=0) - tmp_ste/2, np.nanmean(ste_signal, axis=0) + tmp_ste/2])

						# if smooth_signal:
						# 	condition_ste = sp.signal.decimate(condition_ste, smooth_factor, axis=-1)

						if self.linestylemap is None:
							plt.fill_between(list(range(condition_ste.shape[-1])), condition_ste[0], condition_ste[1], alpha=0.1)
						else:
							plt.fill_between(list(range(condition_ste.shape[-1])), condition_ste[0], condition_ste[1], alpha=0.1, color=self.linestylemap[label][0])		

					if with_stats:
						# extract_data = np.array([np.array(data[key]) for key in conditions])
						# embed()

						if isinstance(sig_marker_ypos, dict):
							ypos = sig_marker_ypos[label]
						elif isinstance(sig_marker_ypos, list):
							ypos = sig_marker_ypos[conditions.index(label)]
						elif sig_marker_ypos is not None:
							ypos = sig_marker_ypos
						else:
							if y_lim[0] is not None:
								ypos = y_lim[0]
							else:
								ypos = 0.0
											
						# Add some jitter to ypos to differentiate multiple conditions
						if jitter_sig_markers:
							ypos = ypos + (0.02*np.random.rand()-0.01)									

						t,p = sp.stats.ttest_1samp(signal, stats_ttest_ref)

						first_sig_pval = True

						for time_point,pval in enumerate(p):

							if (x_lim[0] is not None) and (time_point < x_lim[0]):
								continue

							if (x_lim[1] is not None) and (time_point > x_lim[1]):
								break

							if pval < stats_ttest_p:
								if report_pvals:
									print('found sig. pval (%f) at %i'%(pval,time_point))

								if first_sig_pval and mark_first_sig:
									first_sig_pval = False
									self.vline(time_point, label='', linestyle='solid')

								if self.linestylemap is None:
									plt.text(time_point, ypos,'*', {'color':'k', 'fontsize':16}, alpha = 0.5, horizontalalignment='center', verticalalignment='center')					
								else:
									plt.text(time_point, ypos,'*', {'color':self.linestylemap[label][0],'fontsize':16}, horizontalalignment='center', verticalalignment='center')


					if compute_mean:
						msignal = np.nanmean(signal, axis=0)
					else:
						msignal = signal

					if after_smooth:
						msignal = sp.signal.savgol_filter(msignal, after_smooth_window, 2)						

					if dt:
						dt_msignal = np.diff(msignal, axis=-1) + 0.5

						self.plot(xtimes[:-1], dt_msignal)

						# if not signal_labels:
						# 	if self.linestylemap is None:
						# 		self.plot(xtimes[:-1], dt_msignal, label=label, alpha=1)
						# 	else:
						# 		self.plot(xtimes[:-1], dt_msignal, label=label, alpha=1, color=self.linestylemap[label][0], ls=self.linestylemap[label][1], marker=self.linestylemap[label][2], markersize=MARKERSIZE, markeredgecolor=self.linestylemap[label][3], markerfacecolor=self.linestylemap[label][4], linewidth=self.linestylemap[label][5])
						# else:
						# 	if self.linestylemap is None:
						# 		self.plot(xtimes[:-1], dt_msignal, label=signal_labels[label], alpha=1)
						# 	else:
						# 		self.plot(xtimes[:-1], dt_msignal, label=signal_labels[label], alpha=1, color=self.linestylemap[label][0], ls=self.linestylemap[label][1], marker=self.linestylemap[label][2], markersize=MARKERSIZE, markeredgecolor=self.linestylemap[label][3], markerfacecolor=self.linestylemap[label][4], linewidth=self.linestylemap[label][5])


					if not signal_labels:
						if self.linestylemap is None:
							self.plot(xtimes, msignal, label=label)
						else:
							self.plot(xtimes, msignal, label=label, color=self.linestylemap[label][0], ls=self.linestylemap[label][1], marker=self.linestylemap[label][2], markersize=MARKERSIZE, markeredgecolor=self.linestylemap[label][3], markerfacecolor=self.linestylemap[label][4], linewidth=self.linestylemap[label][5])
					else:
						if self.linestylemap is None:
							self.plot(xtimes, msignal, label=signal_labels[label])
						else:
							self.plot(xtimes, msignal, label=signal_labels[label], color=self.linestylemap[label][0], ls=self.linestylemap[label][1], marker=self.linestylemap[label][2], markersize=MARKERSIZE, markeredgecolor=self.linestylemap[label][3], markerfacecolor=self.linestylemap[label][4], linewidth=self.linestylemap[label][5])



				

		else:
			print('bleh...')
	
					

		plt.ylabel(ylabel)
		plt.xlabel(xlabel)

		if len(title)>0:
			plt.title(title)

		if show_legend:
			if legend_prop is not None:
				plt.legend(loc = legend_loc, prop=legend_prop)
			else:
				plt.legend(loc = legend_loc, fontsize=legend_fontsize)

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

		if axis_ratio is not None:
			plt.axis(axis_ratio)

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
					plt.fill_between(list(range(np.array(reference_mean).size)), condition_mean-condition_ste, condition_mean+condition_ste, alpha=0.5, color=self.linestylemap[key][0])

		
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


	def bar_plot(self, data, conditions, with_error = False, with_stats = False, with_data_points = False, ylabel = '', xlabel = '', yticks = [], xticks = [], yticklabels = [], xticklabels = [], x_lim = [None, None], y_lim = [None, None], output_latex = False, engine = 'seaborn'):
		if self.incorrect_data_format(data, conditions):
			return
		

		latex_code = '\\addplot plot coordinates {'

		opacity = 1

		if self.linestylemap is None:
			# bar_color = {key:'w' for key in conditions}
			# edge_color = {key:'k' for key in conditions}			
			palette = 'viridis'
			saturation = 0.75
			opacity = 1.0
		else:
			palette = {key: self.linestylemap[key][0] for key in conditions}
			saturation = self.linestylemap['saturation']
			opacity = self.linestylemap['opacity']
			# edge_color = {key: [self.linestylemap[key][0][0], self.linestylemap[key][0][1], self.linestylemap[key][0][2], 1.0] for key in conditions}
			# opacity = 0.7
			#bar_opa = {key: self.linestylemap[key][0] for key in conditions}

		bar_width = 0.75

		if engine == 'seaborn':

			if not isinstance(data,pd.DataFrame):
				data = pd.DataFrame.from_dict(data)

			if with_error:
				sn.barplot(data=data[conditions], palette=palette, saturation=saturation, ci = 68, n_boot=500)
			else:
				sn.barplot(data=data[conditions], palette=palette, saturation=saturation)

		else:
			for ii,key in enumerate(conditions):

				if with_error:
					plt.bar(ii+1, np.nanmean(data[key]), yerr = np.nanstd(data[key])/np.sqrt(len(data[key])), width = bar_width, linewidth = 2, label = key, color= bar_color[key],edgecolor=edge_color[key])
				else:
					plt.bar(ii+1, np.nanmean(data[key]), width = bar_width, linewidth = 2, label = key, color = bar_color[key], edgecolor=edge_color[key])
				
				if with_data_points:
					plt.plot(np.ones((len(data[key]),1))*ii+1, data[key], 'o', color=edge_color[key], edge_color = bar_color[key])

				latex_code += '(%i,%.2f) '%(ii, np.mean(data[key]))
		
		latex_code += '};'

		if output_latex:
			print(latex_code)

		plt.ylabel(ylabel)
		plt.xlabel(xlabel)

		if len(xticks)>0:
			if len(xticklabels)>0:
				plt.xticks(xticks, xticklabels)
			else:
				plt.xticks(xticks)
		else:
			if len(xticklabels)>0:
				plt.xticks(1+np.arange(len(conditions)), xticklabels)
			else:
				plt.xticks(1+np.arange(len(conditions)), conditions)

		if len(yticks)>0:
			if len(yticklabels)>0:
				plt.yticks(yticks, yticklabels)
			else:
				plt.yticks(yticks)			

		plt.xlim(x_lim[0], x_lim[1])
		plt.ylim(y_lim[0], y_lim[1])

		# plt.axis('square')

		# sn.set({'ytick.color':[.5,.5,.5]})

		sn.despine(offset={'left':5,'bottom':5}, trim=True)

	def violinplot(self, data, y_lim = [None,None], x_lim = [None, None], xticklabels = [], leftOn=True, bottomOn=True):

		if not isinstance(data, pd.DataFrame):
			data = pd.DataFrame.from_dict(data)

		if self.linestylemap is None:
			sn.violinplot(data=data)		
		else:
			palette = {key: self.linestylemap[key][0] for key in data.keys()}
			sn.violinplot(data=data, palette=palette, saturation = self.linestylemap['saturation'])

		plt.xlim(x_lim[0], x_lim[1])
		plt.ylim(y_lim[0], y_lim[1])

		if len(xticklabels)>0:
			plt.xticks(np.arange(len(data.keys())), xticklabels)

		sn.despine(left=leftOn, bottom=bottomOn, offset={'left':5,'bottom':0}, trim=True)

	def factor_plot(data, *args, **kwargs):
		# Draw a nested barplot to show survival for class and sex
		g = sn.factorplot(data=data, *args, **kwargs)
		g.despine(left=True)

	def hline(self, y = 0, color='k', linewidth = 0.75, linestyle='dashed', alpha=0.5, label=None):
		
		if label is not None:
			plt.text(0.55, y, label, alpha = 0.5, fontsize=8, horizontalalignment='left', verticalalignment='center', bbox=dict(facecolor='w',edgecolor='w'))

		plt.axhline(y = y, color=color, linewidth = linewidth, figure=self.figure, linestyle=linestyle, alpha=alpha)	

	def vline(self, x = 0, color='k', linewidth = 0.75, linestyle='dashed', alpha=0.5, label=None):
		plt.axvline(x = x, color=color, linewidth = linewidth, figure=self.figure, linestyle=linestyle, alpha=alpha)

		if label is not None:
			plt.text(x, 0.55, label, alpha = 0.5, fontsize=8, horizontalalignment='left', verticalalignment='center', bbox=dict(facecolor='w',edgecolor='w'))

	def show(self):

		try:
			self.figure.tight_layout()
		except:
			pass

		plt.show()

	def save_figure(self, filename = '', sub_folder = ''):

		try:
			self.figure.tight_layout()
		except:
			pass

		# Create a random PDF filename if none is provided
		if len(filename)==0: 
			filename = "".join(np.random.choice(alphabetnum, [1, 8])) + '.pdf'

		if len(sub_folder) > 0:
			if not os.path.isdir(os.path.join(self.figure_folder, sub_folder+'/')):
				os.makedirs(os.path.join(self.figure_folder, sub_folder+'/'))

			self.figure.savefig(os.path.join(self.figure_folder, sub_folder, filename))
		else:
			self.figure.savefig(os.path.join(self.figure_folder, filename))