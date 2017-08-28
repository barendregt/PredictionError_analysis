

import numpy as np
import scipy as sp

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
from BehaviorAnalyzer import BehaviorAnalyzer
from Plotter import Plotter


from analysis_parameters import *

pl = Plotter(figure_folder = figfolder, linestylemap=linestylemap)

# raw_data_folder = '/home/barendregt/Projects/PredictionError/fMRI/Attention_Prediction/data/'

# shared_data_folder = '/home/barendregt/Projects/PredictionError/fMRI/Attention_Prediction/data/'

# sublist = ['mb1','mb4']

all_rts = []			 
rts = {'PP': [],
	 'UP': [],
	 'PU': [],
	 'UU': []}

for subname in sublist:

	# print subname
	# Organize filenames
	rawfolder = os.path.join(raw_data_folder,subname)
	sharedfolder = os.path.join(shared_data_folder,subname)
	csvfilename = glob.glob(rawfolder + '/*.csv')#[-1]
	h5filename = os.path.join(sharedfolder,subname+'.h5')

	pa = BehaviorAnalyzer(subname, csvfilename, h5filename, rawfolder, reference_phase = 7, signal_downsample_factor = down_fs, signal_sample_frequency = signal_sample_frequency, deconv_sample_frequency = deconv_sample_frequency, deconvolution_interval = response_deconvolution_interval, verbosity = 0)

	pa.load_data()

	sub_rts = pa.compute_reaction_times()

	all_rts.append(sub_rts)

all_rts = pd.concat(all_rts, keys = sublist, names = ['subject','trial'])

all_rts['rt_norm'] = np.zeros((all_rts.shape[0],1))
# embed()
for subname in sublist:
	all_rts.loc[subname]['rt_norm'][np.array(all_rts.loc[subname]['correct']==1,dtype=bool)] = (all_rts.loc[subname]['reaction_time'][np.array(all_rts.loc[subname]['correct']==1,dtype=bool)] / all_rts.loc[subname]['reaction_time'][np.array(all_rts.loc[subname]['correct']==1,dtype=bool)].median()) * all_rts['reaction_time'][np.array(all_rts['correct']==1,dtype=bool)].median()
	all_rts.loc[subname]['rt_norm'][np.array(all_rts.loc[subname]['correct']==0,dtype=bool)] = (all_rts.loc[subname]['reaction_time'][np.array(all_rts.loc[subname]['correct']==0,dtype=bool)] / all_rts.loc[subname]['reaction_time'][np.array(all_rts.loc[subname]['correct']==0,dtype=bool)].median()) * all_rts['reaction_time'][np.array(all_rts['correct']==0,dtype=bool)].median()

all_rts['condition'] = np.zeros((all_rts.shape[0],1))
tc_lookup = [0,10,30,50,70]
conditions = ['PP','UP','PU','UU']
for tii in range(len(tc_lookup)-1):
	all_rts['condition'][(all_rts['trial_code']>=tc_lookup[tii]) * (all_rts['trial_code']<tc_lookup[tii+1])] = conditions[tii]

# embed()
pl.open_figure(force=1)
# import matplotlib.pyplot as plt 
# import seaborn as sn 

# sn.factorplot(data=all_rts, x="condition", y="reaction_time", hue="correct", size=6, kind="bar", palette="muted")
avg_rts = {}
avg_rts['PP'] = []
avg_rts['PU'] = []
avg_rts['UP'] = []
avg_rts['UU'] = []

for subname in sublist:
	avg_rts['PP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==1)]))
	avg_rts['PU'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UP') * (all_rts.loc[subname]['correct']==1)]))
	avg_rts['UP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PU') * (all_rts.loc[subname]['correct']==1)]))
	avg_rts['UU'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UU') * (all_rts.loc[subname]['correct']==1)]))


# avg_rts['PP'] = np.array(avg_rts['PP'])
# avg_rts['UP'] = np.array(avg_rts['UP']) + 0.14
# avg_rts['UU'] = np.array(avg_rts['UU']) + 0.11
# avg_rts['PU'] = np.array(avg_rts['PU'])


# avg_rts['UP'] = avg_rts['UP'] / avg_rts['PP'].mean()
# avg_rts['PU'] = avg_rts['PU'] / avg_rts['PP'].mean()
# avg_rts['UU'] = avg_rts['UU'] / avg_rts['PP'].mean()

pl.bar_plot(data = avg_rts, conditions = ['PP','UP','PU','UU'], with_error = True, ylabel='Reaction time (s)')

pl.save_figure('rt_correct_raw.pdf',sub_folder='over_subs/task')


pl.open_figure(force=1)
avg_rts = {}
avg_rts['PP'] = []
avg_rts['PU'] = []
avg_rts['UP'] = []
avg_rts['UU'] = []

for subname in sublist:
	# avg_rts['PP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
	avg_rts['PU'].append(np.mean(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UP') * (all_rts.loc[subname]['correct']==1)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==1)])))
	avg_rts['UP'].append(np.mean(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PU') * (all_rts.loc[subname]['correct']==1)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==1)])))
	avg_rts['UU'].append(np.mean(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UU') * (all_rts.loc[subname]['correct']==1)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==1)])))


pl.bar_plot(data = avg_rts, conditions = ['UP','PU','UU'], with_error = True, ylabel='Reaction time (% of predicted)', y_lim = [1.0, 1.2])

# pl.bar_plot
pl.save_figure('rt_correct_norm.pdf',sub_folder='over_subs/task')



pl.open_figure(force=1)
avg_rts = {}
avg_rts['PP'] = []
avg_rts['PU'] = []
avg_rts['UP'] = []
avg_rts['UU'] = []

for subname in sublist:
	avg_rts['PP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
	avg_rts['PU'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PU') * (all_rts.loc[subname]['correct']==0)]))
	avg_rts['UP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UP') * (all_rts.loc[subname]['correct']==0)]))
	avg_rts['UU'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UU') * (all_rts.loc[subname]['correct']==0)]))

avg_rts['PP'] = np.array(avg_rts['PP'])
avg_rts['UP'] = np.array(avg_rts['UP']) 
avg_rts['UU'] = np.array(avg_rts['UU'])
avg_rts['PU'] = np.array(avg_rts['PU'])


pl.bar_plot(data = avg_rts, conditions = ['PP','UP','PU','UU'], with_error = True, ylabel='Reaction time (s)')

# pl.bar_plot
pl.save_figure('rt_incorrect_raw.pdf',sub_folder='over_subs/task')

pl.open_figure(force=1)
avg_rts = {}
avg_rts['PP'] = []
avg_rts['PU'] = []
avg_rts['UP'] = []
avg_rts['UU'] = []

for subname in sublist:
	# avg_rts['PP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
	avg_rts['PU'].extend(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PU') * (all_rts.loc[subname]['correct']==0)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
	avg_rts['UP'].extend(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UP') * (all_rts.loc[subname]['correct']==0)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
	avg_rts['UU'].extend(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UU') * (all_rts.loc[subname]['correct']==0)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))


pl.bar_plot(data = avg_rts, conditions = ['UP','PU','UU'], with_error = True, ylabel='Reaction time (% of predicted)', y_lim = [1.0, 1.2])

# pl.bar_plot
pl.save_figure('rt_incorrect_norm.pdf',sub_folder='over_subs/task')

# embed()