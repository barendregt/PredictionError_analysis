from __future__ import division

import numpy as np
import scipy as sp

import matplotlib
# matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt

from math import *
import os,glob,sys,platform,time

from collections import defaultdict

import pickle
import pandas as pd

from IPython import embed

import hddm
import kabuki

from analysis_parameters import *

figure_dir = 'hddm/figures'

model_dir = 'hddm/models'

# Z models
model_list = [{'name':'RT_and_pupil-vzt_4482',
					'parameters':['v','z','t'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},

			  {'name':'RT_and_pupil-vzt_6514',
					'parameters':['v','z','t'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},
			  {'name':'RT_and_pupil-vzt_9613',
					'parameters':['v','z','t'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},					
			]

# V and T models
model_list = [{'name':'RT_and_pupil-v_and_t-hdf_0',
					'parameters':['v','t'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},

			  {'name':'RT_and_pupil-v_and_t-hdf_1',
					'parameters':['v','t'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},
			  {'name':'RT_and_pupil-v_and_t-hdf_2',
					'parameters':['v','t'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},	
			   {'name':'RT_and_pupil-v_and_t_and_z-hdf_0',
					'parameters':['v','t'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},

			  {'name':'RT_and_pupil-v_and_t_and_z-hdf_1',
					'parameters':['v','t'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},
			  {'name':'RT_and_pupil-v_and_t_and_z-hdf_2',
					'parameters':['v','t'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},									
			]			



def collect_traces(models, param_list = [], store=False):
	# Extract the traces (post. distr.) from a list of models
	#
	# Assumes each model has the same parameters and the same
	# ordering of covariates (if applicable)
	#
	# Returns a dictionary with concatenated traces per parameter

	# Build dict for traces
	trace_col = {p: [] for p in param_list}

	# Collect traces from models
	for model in models:
		nodes, _ = extract_node_params(model, nodes_list = model['conditions_of_interest'])
		
		[trace_col[p].append([subnode.trace()[:] for subnode in nodes[p]]) for p in param_list]

	# Concatenate traces per parameter/covariate
	# (only if there is more than 1 model, to save time)
	if len(model_list) > 1:
		for p in trace_col.keys():
			trace_col[p] = np.hstack(trace_col[p])


	if not(store==False):

		filename = os.path.join(model_dir,'traces',store,'.db')

		with open(filename,mode='w') as f:
			pickle.dump(trace_col, f)
			f.close()

	return trace_col

def load_traces(filename):

	return pickle.load(os.path.join(model_dir,'traces',filename))


def extract_node_params(model, nodes_list = []):

	mdl = hddm.load('hddm/models/%s'%model['name'])

	nodes = defaultdict()

	for param in model['parameters']:
		nodes[param] = list()

		for node_type, int_term in zip(nodes_list, model['interact_term']):
			nodes[param].append(mdl.nodes_db.node['%s_%s[%s]%s'%(param,model['poi_id'],node_type,int_term)])
		
	return nodes, mdl

def plot_posteriors(traces={}, param_list=[], condition_list = [], condition_colors = [], legend_labels = [], filled = True, bins=20, show=True, save=False):

	# Figure out plotting layout
	if len(param_list) < 4:
		sub1 = 1
		sub2 = len(param_list)
	elif len(param_list) < 9:
		sub1 = 2
		sub2 = ceil(len(param_list)/2)
	else:
		sub1 = 3
		sub2 = ceil(len(param_list)/3)

	if len(legend_labels)==0:
		legend_labels = condition_list

	# Plot distributions
	plt.figure()

	for i,param in enumerate(param_list):
		plt.subplot(sub1,sub2,i+1)
		plt.title(parameter_map[param])

		for i,con in enumerate(condition_list):

			x_range = [min(traces[param][i]) - (min(traces[param][i])*0.1), max(traces[param][i]) + (max(traces[param][i])*0.1)]

			x = np.linspace(x_range[0], x_range[1], 300)
			x_histo = np.linspace(x_range[0], x_range[1], bins)
			histo = np.histogram(traces[param][i], bins=bins, range=x_range, density=True)[0]
			interp = sp.interpolate.InterpolatedUnivariateSpline(x_histo, histo)(x)
			interp[interp<0] = 0 # Kick out negative values from interpolation

			if filled:
				plt.fill_between(x, 0, interp, where=interp>0, alpha=0.5)#, facecolor=linestylemap[con][0])
			else:
				plt.plot(x, interp)#, color=linestylemap[con][0], ls=linestylemap[con][1], lw=linestylemap[con][-1])

		plt.xlabel(param)
		plt.ylabel('Posterior probability')
		plt.legend(legend_labels, loc='best')

	if show:
		plt.show()

	if save:
		plt.savefig(os.path.join(figure_dir,'posttraces-%s.pdf'%(time.ctime().replace(' ','_').replace(':',''))))
		plt.close()	

def stat_test_posteriors(traces, trace_labels = []):

	# Test each trace against each other one 
	# Just pick the one you're interested in afterwards....

	if len(trace_labels)==0:
		trace_labels = np.arange(len(traces))

	trace_stats = np.zeros((len(traces),len(traces)),dtype=float)#{key: {key: [] for key in trace_labels} for key in trace_labels}

	for i,ti in enumerate(traces):
		for j,tj in enumerate(traces):
			trace_stats[i][j] = (ti<tj).mean()

	return trace_stats

parameter_map = {'a':'threshold','v': 'drift-rate','t':'nd_time','z':'bias'}

traces = collect_traces(model_list, ['z','v','t'],store='rt_pupil_vt')

# Apply custom link function to compute z-param
traces['z'] = 1/(1+np.exp(-traces['z']))

plot_posteriors(traces, ['v','t','z'], ['PP','UU','UP','PU'],legend_labels=['PP','UU','UP','PU'])

embed()



