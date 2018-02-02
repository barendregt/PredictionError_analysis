import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt

from math import *
import os,glob,sys,platform,time

import pp

from collections import defaultdict

import pickle
import pandas as pd

from IPython import embed

import hddm
import kabuki

figure_dir = 'hddm/figures'

model_dir = 'hddm/models'

model_list = [{'name':'only_RT-only_v',
					'parameters':['v'],
					'poi_id':'C(trial_PE)',
					'conditions_of_interest' : ['PP','UP','PU','UU'],
					'interact_term':['','','','']},
			  {'name':'only_RT-v_and_t',
			  		'parameters':['v','t'],
			  		'poi_id':'C(trial_PE)',
			  		'conditions_of_interest' : ['PP','UP','PU','UU'],
			  		'interact_term':['','','','']},
			  {'name':'RT_and_pupil-only_v',
			  		'parameters':['v'],
			  		'poi_id':'C(trial_PE)',
			  		'conditions_of_interest' : ['PP','UP','PU','UU'],
			  		'interact_term':['','','','']},			  		
			  		# 'conditions_of_interest' : ['PP','UP','PU','UU','T.UP','T.PU','T.UU'],
			  		# 'interact_term':['','','','',':pupil_response_1',':pupil_response_1',':pupil_response_1']},
			  {'name':'RT_and_pupil_2-only_v',
			  		'parameters':['v'],
			  		'poi_id':'C(trial_PE)',
			  		'conditions_of_interest' : ['PP','UP','PU','UU'],
			  		'interact_term':['','','','']},			  		
			  		# 'conditions_of_interest' : ['PP','UP','PU','UU','T.UP','T.PU','T.UU'],
			  		# 'interact_term':['','','','',':pupil_response_2',':pupil_response_2',':pupil_response_2']},
			  		]


def plot_posteriors(nodes, model_name = 'model', param_name = 'param', legend_labels = [], show = False, save = True):

	hddm.analyze.plot_posterior_nodes(nodes)
	plt.xlabel(param_name)
	plt.ylabel('Posterior probability')

	if len(legend_labels)==len(nodes):
		plt.legend(legend_labels)

	if show:
		plt.show()

	if save:
		plt.savefig(os.path.join(figure_dir,'%s.%s.pdf'%(model_name,param_name)))

	plt.close()


def extract_node_params(model, nodes_list = []):

	mdl = hddm.load('hddm/models/%s_combined'%model['name'])

	nodes = defaultdict()

	for param in model['parameters']:
		nodes[param] = list()

		for node_type, int_term in zip(nodes_list, model['interact_term']):
			nodes[param].append(mdl.nodes_db.node['%s_%s[%s]%s'%(param,model['poi_id'],node_type,int_term)])
		
	return nodes


parameter_map = {'a':'threshold','v': 'drift-rate','t':'nd_time','z':'bias'}



for model in model_list:
	nodes = extract_node_params(model, nodes_list = model['conditions_of_interest'])

	[plot_posteriors(nodes[p], model_name=model['name'], param_name=parameter_map[p], legend_labels=model['conditions_of_interest']) for p in model['parameters']]


embed()