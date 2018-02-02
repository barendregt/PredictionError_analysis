import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt

from math import *
import os,glob,sys,platform,time

import pp

import pickle
import pandas as pd

from IPython import embed

import hddm
import kabuki

figure_dir = 'hddm/figures'

model_dir = 'hddm/models'
#model_name = 'single_con-no_pupil'
#model_name = 'single_con-reg_pupil'
# model_name = 'single_con-reg_pupil2'
model_name = 'single_con-reg_pupil_1-fixedPP'



model = hddm.load('hddm/models/%s_combined'%model_name)

parameter_map = {'a':'threshold'}#,'v': 'drift-rate','t':'nd_time'}

for poi, pname in parameter_map.iteritems():

	p_int, p_PP, p_PU, p_UP, p_UU = model.nodes_db.ix[["%s_Intercept"%poi,
	                                              	   "%s_pupil_response_1:C(trial_PE, Treatment('PP'))[PP]"%poi,
	                                                   "%s_pupil_response_1:C(trial_PE, Treatment('PP'))[PU]"%poi,
	                                                   "%s_pupil_response_1:C(trial_PE, Treatment('PP'))[UP]"%poi,
	                                                   "%s_pupil_response_1:C(trial_PE, Treatment('PP'))[UU]"%poi], 'node']
	hddm.analyze.plot_posterior_nodes([p_int, p_PP, p_PU, p_UP, p_UU])
	plt.xlabel('drift-rate')
	plt.ylabel('Posterior probability')

	#plt.show()
	plt.savefig(os.path.join(figure_dir,'%s.%s.pdf'%(model_name,pname)))
embed()