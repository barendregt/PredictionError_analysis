from __future__ import division

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

import cPickle as pickle
import pandas as pd

from IPython import embed
from BehaviorAnalyzer import BehaviorAnalyzer
from Plotter import Plotter


from analysis_parameters import *

pl = Plotter(figure_folder = figfolder, linestylemap=linestylemap)



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

	sub_rts = pa.compute_reaction_times()

	all_rts.append(sub_rts)

all_rts = pd.concat(all_rts, keys = sublist, names = ['subject','trial'])

all_rts['rt_norm'] = np.zeros((all_rts.shape[0],1))
# embed()
for subname in sublist:
	all_rts.loc[subname]['rt_norm'][np.array(all_rts['correct']==1,dtype=int)] = (all_rts.loc[subname]['reaction_time'][np.array(all_rts['correct']==1,dtype=int)] / all_rts.loc[subname]['reaction_time'][np.array(all_rts['correct']==1,dtype=int)].median()) * all_rts['reaction_time'][np.array(all_rts['correct']==1,dtype=int)].median()
	all_rts.loc[subname]['rt_norm'][np.array(all_rts['correct']==0,dtype=int)] = (all_rts.loc[subname]['reaction_time'][np.array(all_rts['correct']==0,dtype=int)] / all_rts.loc[subname]['reaction_time'][np.array(all_rts['correct']==0,dtype=int)].median()) * all_rts['reaction_time'][np.array(all_rts['correct']==0,dtype=int)].median()

all_rts['condition'] = np.zeros((all_rts.shape[0],1))
tc_lookup = [0,10,30,50,70]
for tii in range(len(tc_lookup)-1):
	all_rts['condition'][(all_rts['trial_code']>=tc_lookup[tii]) * (all_rts['trial_code']<tc_lookup[tii+1])] = tii

# embed()
# pl.open_figure(force=1)
import matplotlib.pyplot as plt 
import seaborn as sn 

# sn.factorplot(data=all_rts, x="condition", y="reaction_time", hue="correct", size=6, kind="bar", palette="muted")



plt.savefig(os.path.join(figfolder,'over_subs','task','rt_factor_test5.pdf'))
# embed()