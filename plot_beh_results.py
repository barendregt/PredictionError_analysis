import numpy as np
import scipy as sp
import seaborn as sn
sn.set(style='ticks')
import matplotlib.pyplot as pl
# %matplotlib inline
from math import *
import os,glob,datetime

import pickle as pickle
import pandas as pd

from BehaviorAnalyzer import BehaviorAnalyzer
from IPython import embed



raw_data_folder = '/home/barendregt/Projects/Attention_Prediction/Psychophysics/Data' #raw_data'
shared_data_folder = raw_data_folder #'raw_data'
figfolder = '/home/barendregt/Analysis/Attention_Prediction/Figures'

sublist = ['s6']#['s1','s2','s3','s4','s5','s6']#['s1','s2','s4']
down_fs = 100
winlength = 6500
minwinlength = 6000

def sigmoid(x, a, b, l):
	g = 0.5
	# l = 0.1
	# l=0
	return g + (1-g-l)/(1+np.exp(-(x-a)/b))
 
for subii,subname in enumerate(sublist):
	

	# embed()
	rawfolder = os.path.join(raw_data_folder,subname)
	
	tmp = pickle.load(open(os.path.join(rawfolder,'behavior_' + subname + '_databin.p'),'rb'))

	task_data = [0]
	events = tmp[1]
	task_performance = tmp[2]
	# trial_signals = tmp[3]


	colors = ['k','k','b','b']#['k','r','g','b']

	f = pl.figure()

	s = f.add_subplot(221)

	s.set_title('Color expectation')



	# for ii in np.unique(task_data['coded_trials']):
	labels = ['Expected','Unexpected']
	colors = ['k',[.6, .6, .6]]
	lstyles = ['solid','solid']

	for ii in range(len(task_performance['col_pred-v-unpred'])):

		xs = np.linspace(min(task_performance['col_pred-v-unpred'][0][:,0]),
					     max(task_performance['col_pred-v-unpred'][0][:,0]),1000)	

		pl.semilogx(task_performance['col_pred-v-unpred'][ii][:,0],
				task_performance['col_pred-v-unpred'][ii][:,1],
				'o',
				color=sn.desaturate(colors[ii],1),
				alpha = 0.25)
		
		pl.semilogx(xs, 
			    sigmoid(xs, task_performance['psych_curve_params']['col_pred-v-unpred'][ii][0], task_performance['psych_curve_params']['col_pred-v-unpred'][ii][1], task_performance['psych_curve_params']['col_pred-v-unpred'][ii][2]), 
			    color=sn.desaturate(colors[ii],1), 
			    linestyle=lstyles[ii],
			    alpha=1,
			    label=labels[ii])


	# pl.xlabel('Intensity difference (stim2-stim1)')
	pl.ylabel('Performance (% correct)')

	pl.ylim([.5, 1])
	pl.xlim([min(task_performance['col_pred-v-unpred'][0][:,0]), max(task_performance['col_pred-v-unpred'][0][:,0])])

	pl.xticks(task_performance['col_pred-v-unpred'][0][:,0], [0.25,0.35,0.5, 0.7, 1.0, 1.41, 2.0, 2.83, 4])

	pl.legend(loc = 'upper left')
	sn.despine(offset=5)	


	s = f.add_subplot(222)

	s.set_title('Color attention')


	# for ii in np.unique(task_data['coded_trials']):
	labels = ['P-UP','UP-P','UP-UP']
	colors = ['r','g','b']
	lstyles = ['solid','solid','solid']

	for ii in range(len(task_performance['col_att-v-unatt'])):

		xs = np.linspace(min(task_performance['col_pred-v-unpred'][0][:,0]),
					     max(task_performance['col_pred-v-unpred'][0][:,0]),1000)	

		pl.semilogx(xs, 
			    sigmoid(xs, task_performance['psych_curve_params']['col_pred-v-unpred'][0][0], task_performance['psych_curve_params']['col_pred-v-unpred'][0][1], task_performance['psych_curve_params']['col_pred-v-unpred'][0][2]), 
			    color='k', 
			    linestyle=lstyles[0],
			    alpha=0.75,
			    label='base')		

		pl.semilogx(task_performance['col_att-v-unatt'][ii][:,0],
				task_performance['col_att-v-unatt'][ii][:,1],
				'o',
				color=sn.desaturate(colors[ii],1),
				alpha = 0.25)
		
		pl.semilogx(xs, 
			    sigmoid(xs, task_performance['psych_curve_params']['col_att-v-unatt'][ii][0], task_performance['psych_curve_params']['col_att-v-unatt'][ii][1], task_performance['psych_curve_params']['col_att-v-unatt'][ii][2]), 
			    color=sn.desaturate(colors[ii],1), 
			    linestyle=lstyles[ii],
			    alpha=1,
			    label=labels[ii])


	# pl.xlabel('Intensity difference (stim2-stim1)')
	# pl.ylabel('Performance (% correct)')
	pl.ylim([.5, 1])
	pl.xlim([min(task_performance['col_pred-v-unpred'][0][:,0]), max(task_performance['col_pred-v-unpred'][0][:,0])])
	pl.xticks(task_performance['col_pred-v-unpred'][0][:,0], [0.25,0.35,0.5, 0.7, 1.0, 1.41, 2.0, 2.83, 4])

	pl.legend(loc='upper left')
	sn.despine(offset=5)		

	s = f.add_subplot(223)

	s.set_title('Ori expectation')



	# for ii in np.unique(task_data['coded_trials']):
	labels = ['Expected','Unexpected']
	colors = ['k',[.6, .6, .6]]
	lstyles = ['solid','solid']

	for ii in range(len(task_performance['ori_pred-v-unpred'])):

		xs = np.linspace(min(task_performance['ori_pred-v-unpred'][0][:,0]),
					     max(task_performance['ori_pred-v-unpred'][0][:,0]),1000)	

		pl.semilogx(task_performance['ori_pred-v-unpred'][ii][:,0],
				task_performance['ori_pred-v-unpred'][ii][:,1],
				'o',
				color=sn.desaturate(colors[ii],1),
				alpha = 0.25)
		
		pl.semilogx(xs, 
			    sigmoid(xs, task_performance['psych_curve_params']['ori_pred-v-unpred'][ii][0], task_performance['psych_curve_params']['ori_pred-v-unpred'][ii][1], task_performance['psych_curve_params']['ori_pred-v-unpred'][ii][2]), 
			    color=sn.desaturate(colors[ii],1), 
			    linestyle=lstyles[ii],
			    alpha=1,
			    label=labels[ii])


	pl.xlabel('Intensity difference (Quest units)')
	pl.ylabel('Performance (% correct)')

	pl.ylim([.5, 1])
	pl.xlim([min(task_performance['ori_pred-v-unpred'][0][:,0]), max(task_performance['ori_pred-v-unpred'][0][:,0])])
	pl.xticks(task_performance['ori_pred-v-unpred'][0][:,0], [0.25,0.35,0.5, 0.7, 1.0, 1.41, 2.0, 2.83, 4])

	# pl.legend()
	sn.despine(offset=5)	


	s = f.add_subplot(224)

	s.set_title('Ori attention')


	# for ii in np.unique(task_data['coded_trials']):
	labels = ['P-UP','UP-P','UP-UP']
	colors = ['r','g','b']
	lstyles = ['solid','solid','solid']

	for ii in range(len(task_performance['ori_att-v-unatt'])):

		xs = np.linspace(min(task_performance['ori_pred-v-unpred'][0][:,0]),
					     max(task_performance['ori_pred-v-unpred'][0][:,0]),1000)	

		pl.semilogx(xs, 
			    sigmoid(xs, task_performance['psych_curve_params']['ori_pred-v-unpred'][0][0], task_performance['psych_curve_params']['ori_pred-v-unpred'][0][1], task_performance['psych_curve_params']['ori_pred-v-unpred'][0][2]), 
			    color='k', 
			    linestyle=lstyles[0],
			    alpha=0.75,
			    label='base')

		pl.semilogx(task_performance['ori_att-v-unatt'][ii][:,0],
				task_performance['ori_att-v-unatt'][ii][:,1],
				'o',
				color=sn.desaturate(colors[ii],1),
				alpha = 0.25)
		
		pl.semilogx(xs, 
			    sigmoid(xs, task_performance['psych_curve_params']['ori_att-v-unatt'][ii][0], task_performance['psych_curve_params']['ori_att-v-unatt'][ii][1], task_performance['psych_curve_params']['ori_att-v-unatt'][ii][2]), 
			    color=sn.desaturate(colors[ii],1), 
			    linestyle=lstyles[ii],
			    alpha=1,
			    label=labels[ii])


	pl.xlabel('Intensity difference (Quest units)')
	# pl.ylabel('Performance (% correct)')
	pl.ylim([.5, 1])
	pl.xlim([min(task_performance['ori_pred-v-unpred'][0][:,0]), max(task_performance['ori_pred-v-unpred'][0][:,0])])
	pl.xticks(task_performance['ori_pred-v-unpred'][0][:,0], [0.25,0.35,0.5, 0.7, 1.0, 1.41, 2.0, 2.83, 4])

	# pl.legend()
	sn.despine(offset=5)		

	pl.tight_layout()

	f.savefig(os.path.join(figfolder, subname + '_behavior' + '.pdf'))# + '-' + str(datetime.datetime.now()) + '.pdf'))
	# # f.close()
	pl.close()
	# embed()