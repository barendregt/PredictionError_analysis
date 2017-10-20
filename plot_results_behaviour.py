

import numpy as np
import scipy as sp

import matplotlib.pyplot as mp
import seaborn as sns

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
all_pcs = []
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

	# sub_pcs = pa.compute_percent_correct()

	all_rts.append(sub_rts)

	# all_pcs.append(sub_pcs)




all_rts = pd.concat(all_rts, keys = sublist, names = ['subject','trial'])

for s in sublist:
	all_rts['reaction_time'][s] = all_rts['reaction_time'][s].values / np.mean(all_rts['reaction_time'][s].values)

all_rts['condition'] = 'Predicted'

all_rts.loc[(all_rts['trial_code'] >= 10.0) & (all_rts['trial_code'] <= 20.0),'condition'] = 'Irrelevant'
all_rts.loc[(all_rts['trial_code'] >= 30.0) & (all_rts['trial_code'] <= 40.0),'condition'] = 'Relevant'
all_rts.loc[(all_rts['trial_code'] >= 50.0) & (all_rts['trial_code'] <= 60.0),'condition'] = 'Both'

all_rts['task'] = 'Color'
all_rts.loc[(all_rts['trial_code'] == 1) | ((all_rts['trial_code'] == 20) | ((all_rts['trial_code'] == 40) | (all_rts['trial_code'] == 60))),'task'] = 'Orientation'


# Make some plots to visualize data
sns.set(style="white", context="talk")

f,(ax1,ax2) = mp.subplots(1,2)

g = sns.factorplot(ax=ax1,x="task", y="correct", hue="condition", data=all_rts,
                   size=6, kind="bar", palette="muted",legend=False)
#g.despine(ax=ax1,left=True)
ax1.set_ylabel("proportion correct responses")

ax1.set(ylim=[.5, 1])

ax1.legend(loc='top right', shadow=True, fontsize='small')
ax1.set_title('Task performance')

g = sns.factorplot(ax=ax2,x="task", y="reaction_time", hue="condition", data=all_rts,
                   size=6, kind="bar", palette="muted",legend=False)
#g.despine(ax=ax2,left=True)
ax2.set_ylabel("reaction_time (sec)")

ax2.set(ylim=[0, 2])

ax2.legend(loc='top right', shadow=True, fontsize='small')
ax2.set_title('Reaction time')

sns.despine()
mp.tight_layout()

mp.show()


embed()	