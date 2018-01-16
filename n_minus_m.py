

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

pl = Plotter(figure_folder = figfolder, linestylemap=linestylemap, sn_style='white')

# raw_data_folder = '/home/barendregt/Projects/PredictionError/fMRI/Attention_Prediction/data/'

# shared_data_folder = '/home/barendregt/Projects/PredictionError/fMRI/Attention_Prediction/data/'

# sublist = ['mb1','mb4']

all_rts = []			 
rts_correct = {'PP': [],
	 'UP': [],
	 'PU': [],
	 'UU': []}

rts_incorrect = {'PP': [],
	 'UP': [],
	 'PU': [],
	 'UU': []}	 

pcs = {'PP': [],
	 'UP': [],
	 'PU': [],
	 'UU': []}	 
data = pd.DataFrame()

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

	#sub_pcs = pa.compute_percent_correct()



	for key in list(pcs.keys()):

		sub_rts['trial_code'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1])] = key

		# pcs[key].append(sub_rts['correct'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1])].mean())
		# rts_correct[key].append(sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==1)].median())
		# rts_incorrect[key].append(sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==0)].median())


	#sub_rts['reaction_time'] = sp.signal.detrend(sub_rts['reaction_time'].values)

	sub_rts['next_rt'] = np.hstack([sub_rts['reaction_time'][1:].values,np.NaN])
	sub_rts['next_correct'] = np.hstack([sub_rts['correct'][1:].values,np.NaN])

	sub_rts['subID'] = subname

	data = data.append(sub_rts, ignore_index=False)

	# # data = []
	# for i in range(sub_rts.shape[0]-1):
	# 	if sub_rts['correct'][i]==0:
	# 		data['incorrect'][condition_keymap[sub_rts['trial_code'][i]]].extend(sub_rts['reaction_time'][[i,i+1]])
	# 	else:
	# 		data['correct'][condition_keymap[sub_rts['trial_code'][i]]].extend(sub_rts['reaction_time'][[i,i+1]])

	# data = np.array(data)

	# sn.regplot(x=data[:,0],y=data[:,1])
	# plt.show()
	# r = np.corrcoef(sub_rts)


	#all_rts.append(sub_rts)
# embed()


# kick out last trial of each PP
data = data.drop(298)

# data = np.array(data)

import seaborn as sn
import matplotlib.pyplot as plt

# plt.figure()

g=sn.lmplot(x="reaction_time",y="next_rt",hue="trial_code",col="correct",data=data,col_wrap=2,x_bins=5,truncate=True,aspect=1)

g = (g.set_axis_labels("Reaction time 1-back (s)", "Reaction time current (s)")
	.set(xlim=(0.5,2.0), ylim=(0.5, 2.0),
	xticks=[.5,1.0,1.5,2.0], yticks=[.5,1.0,1.5,2.0])
	)

# plt.show()

plt.savefig(figfolder + '/over_subs/task/n-back_reaction_times.pdf')

# pd_rts = pd.DataFrame()

# for key in rts_correct.keys():
# 	tmp = {}
# 	tmp['Reaction time'] = np.hstack([rts_correct[key], rts_incorrect[key]])
# 	tmp['Response'] = np.squeeze(np.hstack([['Correct']*len(sublist),['Incorrect']*len(sublist)]))
# 	tmp['subID'] = np.squeeze(np.tile(np.arange(len(sublist)),(1,2)))
# 	tmp['prediction_error'] = [keymap_to_words[key]]*2*len(sublist)
# 	tmp['TR'] = keymap_to_code[key][0]
# 	tmp['TI'] = keymap_to_code[key][1]

# 	pd_rts = pd_rts.append(other=pd.DataFrame.from_dict(tmp),ignore_index=True)

# import matplotlib.pyplot as plt
# import seaborn as sn

# palette = {}
# for key in rts_correct.keys():
# 	palette[keymap_to_words[key]] = linestylemap[key][0]
# saturation = linestylemap['saturation']

# plt.figure()

# sn.factorplot(data=pd_rts, x='Response',y='Reaction time',hue='prediction_error',kind='bar', size=10, aspect=1.5, palette=palette,saturation=saturation,ci=68)

# plt.save_figure([figfolder+'/over_subs/task/reaction_times.pdf'])
