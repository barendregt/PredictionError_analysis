import numpy as np
import scipy as sp
import seaborn as sn
sn.set(style='ticks')
import matplotlib.pyplot as pl
# %matplotlib inline
from math import *
import os,glob

import pickle as pickle
import pandas as pd

from BehaviorAnalyzer import BehaviorAnalyzer
from IPython import embed



raw_data_folder = '/home/barendregt/Projects/Attention_Prediction/Psychophysics/Data' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/Attention_Prediction/Figures'

sublist = ['s1','s2','s3','s4','s5']#['s1','s2','s4']
down_fs = 100
winlength = 6500
minwinlength = 6000

def sigmoid(x, a, b, l):
	g = 0.5
	# l = 0.1
	return g + (1-g-l)/(1+np.exp(-(x-a)/b))
 
for subii,subname in enumerate(sublist):
	


	rawfolder = os.path.join(raw_data_folder,subname)
	
	tmp = pickle.load(open(os.path.join(rawfolder,subname + '_databin.p'),'rb'))

	task_data = [0]
	events = tmp[1]
	task_performance = tmp[2]
	trial_signals = tmp[3]



	colors = ['k','k','b','b']#['k','r','g','b']

	f = pl.figure()

	s = f.add_subplot(121)

	s.set_title('Color')


	pl.axvline(0.5, lw=0.5, alpha=0.5, color='r') # cue
	pl.axvline(0.5+1.25, lw=0.5, alpha=0.5, color='r') # task
	pl.axvline(0.5+1.25+1.25+0.5, lw=0.5, alpha=0.5, color='r') # stimulus
	pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')

	sig = trial_signals['pred-col']
	pl.plot(np.linspace(0,minwinlength/1000,np.size(np.mean(sig,0))),
			np.mean(sig,0), 
			color=sn.desaturate('k',1),
			alpha=0.4,
			label='pred')

	# pl.fill_between(np.linspace(0,minwinlength/1000,np.size(np.mean(sig,0))),
	# 				np.mean(sig,0)-(np.std(sig,0)/sqrt(len(sig))), 
	# 				np.mean(sig,0)+(np.std(sig,0)/sqrt(len(sig))),
	# 				color = 'k',
	# 				alpha = 0.1)

	sig = trial_signals['unpred-col']
	pl.plot(np.linspace(0,minwinlength/1000,np.size(np.mean(sig,0))),
			np.mean(sig,0), 
			color=sn.desaturate('k',1),
			alpha=0.4,
			label='unpred',
			linestyle='dashed')

	# pl.fill_between(np.linspace(0,minwinlength/1000,np.size(np.mean(sig,0))),
	# 				np.mean(sig,0)-(np.std(sig,0)/sqrt(len(sig))), 
	# 				np.mean(sig,0)+(np.std(sig,0)/sqrt(len(sig))),
	# 				color = 'k',
	# 				alpha = 0.1)


	# pl.legend()
	sn.despine(offset=5)	

	s = f.add_subplot(122)

	s.set_title('Orientation')


	pl.axvline(0.5, lw=0.5, alpha=0.5, color='r') # cue
	pl.axvline(0.5+1.25, lw=0.5, alpha=0.5, color='r') # task
	pl.axvline(0.5+1.25+1.25+0.5, lw=0.5, alpha=0.5, color='r') # stimulus
	pl.axhline(0, lw=0.25, alpha=0.5, color = 'k')

	sig = trial_signals['pred-ori']
	pl.plot(np.linspace(0,minwinlength/1000,np.size(np.mean(sig,0))),
			np.mean(sig,0), 
			color=sn.desaturate('b',1),
			alpha=0.4,
			label='pred')

	# pl.fill_between(np.linspace(0,minwinlength/1000,np.size(np.mean(sig,0))),
	# 				np.mean(sig,0)-(np.std(sig,0)/sqrt(len(sig))), 
	# 				np.mean(sig,0)+(np.std(sig,0)/sqrt(len(sig))),
	# 				color = 'b',
	# 				alpha = 0.1)

	sig = trial_signals['unpred-ori']
	pl.plot(np.linspace(0,minwinlength/1000,np.size(np.mean(sig,0))),
			np.mean(sig,0), 
			color=sn.desaturate('b',1),
			alpha=0.4,
			label='unpred',
			linestyle='dashed')

	# pl.fill_between(np.linspace(0,minwinlength/1000,np.size(np.mean(sig,0))),
	# 				np.mean(sig,0)-(np.std(sig,0)/sqrt(len(sig))), 
	# 				np.mean(sig,0)+(np.std(sig,0)/sqrt(len(sig))),
	# 				color = 'b',
	# 				alpha = 0.1)
	


	pl.legend()
	sn.despine(offset=5)


	# s = f.add_subplot(223)
	# # s.set_title('Performance')

	# # for ii in np.unique(task_data['coded_trials']):
	# labels = ['pred','unpred','pred','unpred']
	# colors = ['k','k','b','b']
	# lstyles = ['solid','dashed','solid','dashed']

	# for ii in range(3):

	# 	# xs = np.linspace(-max(task_performance['task'][ii][:,0]),
	# 	# 				 max(task_performance['task'][ii][:,0]),
	# 	# 				 1000)	

	# 	xs = np.linspace(-1,1,1000)	

	# 	pl.plot(xs, 
	# 		    sigmoid(xs, task_performance['psych_curve_params']['task'][ii][0], task_performance['psych_curve_params']['task'][ii][1],task_performance['psych_curve_params']['task'][ii][2]), 
	# 		    color=sn.desaturate(colors[ii],1), 
	# 		    linestyle=lstyles[ii],
	# 		    alpha=0.5,
	# 		    label=labels[ii])


	# # pl.legend()
	# sn.despine(offset=5)	

	# pl.show()

	# embed()

	f.savefig(os.path.join(figfolder, subname + '_pupil.pdf'))
	# # f.close()
	pl.close()
