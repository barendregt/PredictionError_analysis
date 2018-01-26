import numpy as np
import scipy as sp

import matplotlib
matplotlib.use('Qt4Agg')

from math import *
import os,glob,sys,platform,time

import pp

import pickle
import pandas as pd

from IPython import embed

import hddm
import kabuki



def fit_hddm_model(trace_id, data, model_dir, model_name, samples=10000):

	import os
	import numpy as np
	import hddm
	

	m = hddm.HDDM(data, p_outlier = 0.05, depends_on = {'z':'trial_PE','v':'trial_PE','t':'trial_PE'})
	m.find_starting_values()
	m.sample(samples, burn=samples/10, thin=2, dbname=os.path.join(model_dir, model_name+ '_db{}'.format(trace_id)), db='pickle')

	return m


def fit_reg_model(trace_id, data, model_dir, model_name, samples=10000):

	import os
	import numpy as np
	import hddm
	from patsy import dmatrix  # for regression only

	m = hddm.HDDMRegressor(data, p_outlier = 0.05, depends_on = {'z':'trial_PE','v':'trial_PE','t':'trial_PE'})
	m.find_starting_values()
	m.sample(samples, burn=samples/10, thin=2, dbname=os.path.join(model_dir, model_name+ '_db{}'.format(trace_id)), db='pickle')

	return m

model_dir = 'hddm/models'
model_name = 'single_con-no_pupil'


filename = 'hddm_dataset'

data = hddm.load_csv('%s.csv'%filename)


use_data = data[(data['rt'] > 0) & (data['rt'] < 4)]
use_data = hddm.utils.flip_errors(use_data)

n_jobs = 3
samples = 5000

job_server = pp.Server(ppservers=(), ncpus=n_jobs)
start_time = time.time()
jobs = [(trace_id, job_server.submit(fit_model,(trace_id, use_data, model_dir, model_name, samples), (), ('hddm',))) for trace_id in range(n_jobs)]

models = []
for s, job in jobs:
    models.append(job())
print "Time elapsed: ", time.time() - start_time, "s"
job_server.print_stats()

# save individual models
for i in range(n_jobs):
    model = models[i]
    model.save(os.path.join(model_dir, '{}_{}'.format(model_name,i)))


# Compute gelman rubic (Rhat) thingy
gr = hddm.analyze.gelman_rubin(models)
text_file = open(os.path.join(model_dir, 'diagnostics', '{}-{}.txt'.format('gelman_rubic',model_name)), 'w')
for p in gr.items():
    text_file.write("%s:%s\n" % p)
text_file.close()

# Save combined model
comb_model = kabuki.utils.concat_models(models)
comb_model.save(os.path.join(model_dir, '{}_{}'.format(model_name,'combined')))
