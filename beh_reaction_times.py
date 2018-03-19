

import numpy as np
import scipy as sp

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
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


all_rts_correct = {'PP': [],
	 'UP': [],
	 'PU': [],
	 'UU': []}

all_rts_incorrect = {'PP': [],
	 'UP': [],
	 'PU': [],
	 'UU': []}	


pcs = {'PP': [],
	 'UP': [],
	 'PU': [],
	 'UU': []}	 



all_pd_rts = pd.DataFrame([],columns=['Reaction time','Response','subID','prediction_error','TR','TI'])

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
		pcs[key].append(sub_rts['correct'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1])].mean())
		rts_correct[key].append(sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==1)].median())
		rts_incorrect[key].append(sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==0)].median())
		# all_rts_correct[key].append(sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==1)])
		# all_rts_incorrect[key].append(sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==0)])

		tmp = pd.DataFrame()
		tmp['Reaction time'] = np.hstack([sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==1)], sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==0)]])
		tmp['Reaction time'] = tmp['Reaction time']
		tmp['Response'] = np.hstack([['Correct']*sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==1)].shape[0],['Incorrect']*sub_rts['reaction_time'][(sub_rts['trial_code']==inverse_keymap[key][0]) + (sub_rts['trial_code']==inverse_keymap[key][1]) + (sub_rts['correct']==0)].shape[0]])
		tmp['subID'] = sublist.index(subname)
		tmp['prediction_error'] = keymap_to_words[key]
		tmp['TR'] = keymap_to_code[key][0]
		tmp['TI'] = keymap_to_code[key][1]		

		all_pd_rts = all_pd_rts.append(other=tmp, ignore_index=True)

	#all_rts.append(sub_rts)
# embed()

pd_rts = pd.DataFrame()

for key in rts_correct.keys():
	tmp = {}
	tmp['Reaction time'] = np.hstack([rts_correct[key], rts_incorrect[key]])
	tmp['Response'] = np.squeeze(np.hstack([['Correct']*len(sublist),['Incorrect']*len(sublist)]))
	tmp['subID'] = np.squeeze(np.tile(np.arange(len(sublist)),(1,2)))
	tmp['prediction_error'] = [keymap_to_words[key]]*2*len(sublist)
	tmp['TR'] = keymap_to_code[key][0]
	tmp['TI'] = keymap_to_code[key][1]

	pd_rts = pd_rts.append(other=pd.DataFrame.from_dict(tmp),ignore_index=True)


pd_pcs = pd.DataFrame()

for key in pcs.keys():
	tmp = {}
	tmp['Percentage correct'] = pcs[key]
	#tmp['Response'] = np.squeeze(np.hstack([['Correct']*len(sublist),['Incorrect']*len(sublist)]))
	tmp['subID'] = np.arange(len(sublist))
	tmp['prediction_error'] = [keymap_to_words[key]]*len(sublist)
	tmp['TR'] = keymap_to_code[key][0]
	tmp['TI'] = keymap_to_code[key][1]

	pd_pcs = pd_pcs.append(other=pd.DataFrame.from_dict(tmp),ignore_index=True)

import matplotlib.pyplot as plt
import seaborn as sn

palette = {}
for key in rts_correct.keys():
	palette[keymap_to_words[key]] = linestylemap[key][0]
saturation = linestylemap['saturation']

# plt.figure()


embed()

f=sn.factorplot(data=pd_rts, x='prediction_error',y='Reaction time',kind='bar', size=10, aspect=1.5, palette=palette,saturation=saturation,ci=68)

plt.savefig(figfolder+'/over_subs/task/reaction_times.pdf')

# plt.figure()

# fig, ax = plt.subplots()

g=sn.factorplot(data=pd_pcs, x='prediction_error',y='Percentage correct',kind='bar', size=10, aspect=1.5, palette=palette,saturation=saturation,ci=68)

g = (g.set_axis_labels("Prediction error", "Percentage correct (%)")
	.set(ylim=(0.5, 1.0),xticks=[],
	yticks=[.5,.6,.7,.8,.9,1.0],yticklabels=['50%','60%','70%','80%','90%','100%'])
	)



# Normalize RTs per subject before combining
overall_mean = all_pd_rts['Reaction time'].mean()

all_pd_rts['Normalized RT'] = all_pd_rts['Reaction time'] / np.repeat(all_pd_rts.groupby('subID')['Reaction time'].median().values,all_pd_rts.groupby('subID').count()['Reaction time'].values)
# all_pd_rts['Normalized RT'] *= overall_mean





sn.factorplot(x="prediction_error",y="Normalized RT",data=all_pd_rts,kind="bar",estimator=np.median)




plt.figure(figsize=(10,8))

plt.subplot(2,1,1)
plt.title('Correct')
# plt.axis([0, 3, 0, 1.65])

plt.subplot(2,1,2)
plt.title('Incorrect')
# plt.axis([0, 3, 0, 1.65])

for key in rts_correct.keys():
	plt.subplot(2,1,1)


	y,x = np.histogram(all_pd_rts['Normalized RT'][(all_pd_rts['prediction_error']==keymap_to_words[key]) * (all_pd_rts['Response']=="Correct")], bins=50)

	x = x[:-1]-np.diff(x)


	alpha,loc,scale = sp.stats.gamma.fit(all_pd_rts['Normalized RT'][(all_pd_rts['prediction_error']==keymap_to_words[key]) * (all_pd_rts['Response']=="Correct")])

	fit_vals = sp.stats.gamma.pdf(x, alpha, loc, scale) * np.max(y)

	plt.plot(x,fit_vals,color=linestylemap[key][0],lw=3)


	# ax1 = sn.distplot(all_pd_rts['Normalized RT'][(all_pd_rts['prediction_error']==keymap_to_words[key]) * (all_pd_rts['Response']=="Correct")],kde=False,color=linestylemap[key][0])#,lw=linestylemap[key][-1],ls=linestylemap[key][1])
	# ax2 = ax1.twinx()
	# sn.distplot(all_pd_rts['Normalized RT'][(all_pd_rts['prediction_error']==keymap_to_words[key]) * (all_pd_rts['Response']=="Correct")],hist=False,fit=sp.stats.gamma,color=linestylemap[key][0],ax=ax2)#,lw=linestylemap[key][-1],ls=linestylemap[key][1], ax=ax)
	# # sn.hist()

	plt.subplot(2,1,2)

	y,x = np.histogram(all_pd_rts['Normalized RT'][(all_pd_rts['prediction_error']==keymap_to_words[key]) * (all_pd_rts['Response']=="Incorrect")], bins=50)

	x = x[:-1]-np.diff(x)

	alpha,loc,scale = sp.stats.gamma.fit(all_pd_rts['Normalized RT'][(all_pd_rts['prediction_error']==keymap_to_words[key]) * (all_pd_rts['Response']=="Incorrect")])

	fit_vals = sp.stats.gamma.pdf(x, alpha, loc, scale) * np.max(y)

	plt.plot(x,fit_vals,color=linestylemap[key][0],lw=3)

	# ax1=sn.distplot(all_pd_rts['Normalized RT'][(all_pd_rts['prediction_error']==keymap_to_words[key]) * (all_pd_rts['Response']=="Incorrect")],kde=False,color=linestylemap[key][0])#,lw=linestylemap[key][-1],ls=linestylemap[key][1])
	# ax2 = ax1.twinx()
	# sn.distplot(all_pd_rts['Normalized RT'][(all_pd_rts['prediction_error']==keymap_to_words[key]) * (all_pd_rts['Response']=="Incorrect")],hist=False,fit=sp.stats.gamma,color=linestylemap[key][0],ax=ax2)#,lw=linestylemap[key][-1],ls=linestylemap[key][1], ax=ax)

sn.despine(offset=5)
plt.tight_layout()
plt.show()










# for patch,label in zip(ax.patches,keymap_to_words.values()) :

# 	patch_x = patch.get_x()
# 	patch_width = patch.get_width()

# 	ax.text(patch_x+patch_width/2, 0, label, {'color':'w','fontsize':24,'family':'Helvetica','va':'bottom','ha':'center','fontweight':'bold'}, rotation=90)


# plt.show()


plt.savefig(figfolder+'/over_subs/task/performance.pdf')

# pl.open_figure(force=1)

# pl.bar_plot(data=pcs, conditions=['PP','UP','PU','UU'], xticks=np.arange(4), xticklabels=['None','TaskRel','TaskIrrel','both'], xlabel='Prediction error', ylabel='Proportion correct', y_lim = [0,1], with_error=True)

# #pl.violinplot(data=pcs,y_lim= [0.5,1.05], xticklabels=['None','TaskRel','TaskIrrel','both'], bottomOn=False)

# pl.save_figure(filename='prop_correct.pdf',sub_folder='over_subs/task')

# pd_rts_correct = pd.DataFrame.from_dict(rts_correct)
# pd_rts_incorrect = pd.DataFrame.from_dict(rts_incorrect)

# pl.open_figure(force=1)

# #pl.bar_plot(data=rts, conditions=['PP','UP','PU','UU'], xticks=np.array([1,2,3,4])-0.75/2, xticklabels=['None','TaskRel','TaskIrrel','both'], xlabel='Prediction error', ylabel='Reaction time (s)', y_lim = [0,1.4], with_error=True)
# pl.subplot(1,2,1)
# # pl.violinplot(data=rts_correct,y_lim= [0,2.3], xticklabels=['None','TaskRel','TaskIrrel','both'], bottomOn=False)
# pl.bar_plot(data=rts_correct, conditions=['PP','UP','PU','UU'], xticks=np.arange(4),  xticklabels=['None','TaskRel','TaskIrrel','both'], xlabel='Prediction error', ylabel='Reaction time (s)', y_lim = [0,1.4], with_error=True)

# pl.subplot(1,2,2)
# # pl.violinplot(data=rts_incorrect,y_lim= [0,2.3], xticklabels=['None','TaskRel','TaskIrrel','both'], bottomOn=False)
# pl.bar_plot(data=rts_incorrect, conditions=['PP','UP','PU','UU'], xticks=np.arange(4),  xticklabels=['None','TaskRel','TaskIrrel','both'], xlabel='Prediction error', ylabel='Reaction time (s)', y_lim = [0,1.4], with_error=True)

# pl.save_figure(filename='reaction_times.pdf',sub_folder='over_subs/task')


# pl.show()

# all_rts = pd.concat(all_rts, keys = sublist, names = ['subject','trial'])

# all_rts['rt_norm'] = np.zeros((all_rts.shape[0],1))
# embed()
# for subname in sublist:
# 	all_rts.loc[subname]['rt_norm'][np.array(all_rts.loc[subname]['correct']==1,dtype=bool)] = (all_rts.loc[subname]['reaction_time'][np.array(all_rts.loc[subname]['correct']==1,dtype=bool)] / all_rts.loc[subname]['reaction_time'][np.array(all_rts.loc[subname]['correct']==1,dtype=bool)].median()) * all_rts['reaction_time'][np.array(all_rts['correct']==1,dtype=bool)].median()
# 	all_rts.loc[subname]['rt_norm'][np.array(all_rts.loc[subname]['correct']==0,dtype=bool)] = (all_rts.loc[subname]['reaction_time'][np.array(all_rts.loc[subname]['correct']==0,dtype=bool)] / all_rts.loc[subname]['reaction_time'][np.array(all_rts.loc[subname]['correct']==0,dtype=bool)].median()) * all_rts['reaction_time'][np.array(all_rts['correct']==0,dtype=bool)].median()

# all_rts['condition'] = np.zeros((all_rts.shape[0],1))
# tc_lookup = [0,10,30,50,70]
# conditions = ['PP','UP','PU','UU']
# for tii in range(len(tc_lookup)-1):
# 	all_rts['condition'][(all_rts['trial_code']>=tc_lookup[tii]) * (all_rts['trial_code']<tc_lookup[tii+1])] = conditions[tii]

# # embed()
# pl.open_figure(force=1)
# # import matplotlib.pyplot as plt 
# # import seaborn as sn 

# # sn.factorplot(data=all_rts, x="condition", y="reaction_time", hue="correct", size=6, kind="bar", palette="muted")
# avg_rts = {}
# avg_rts['PP'] = []
# avg_rts['PU'] = []
# avg_rts['UP'] = []
# avg_rts['UU'] = []

# for subname in sublist:
# 	avg_rts['PP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==1)]))
# 	avg_rts['PU'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UP') * (all_rts.loc[subname]['correct']==1)]))
# 	avg_rts['UP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PU') * (all_rts.loc[subname]['correct']==1)]))
# 	avg_rts['UU'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UU') * (all_rts.loc[subname]['correct']==1)]))


# # avg_rts['PP'] = np.array(avg_rts['PP'])
# # avg_rts['UP'] = np.array(avg_rts['UP']) + 0.14
# # avg_rts['UU'] = np.array(avg_rts['UU']) + 0.11
# # avg_rts['PU'] = np.array(avg_rts['PU'])


# # avg_rts['UP'] = avg_rts['UP'] / avg_rts['PP'].mean()
# # avg_rts['PU'] = avg_rts['PU'] / avg_rts['PP'].mean()
# # avg_rts['UU'] = avg_rts['UU'] / avg_rts['PP'].mean()

# pl.bar_plot(data = avg_rts, conditions = ['PP','UP','PU','UU'], with_error = True, ylabel='Reaction time (s)')

# pl.save_figure('rt_correct_raw.pdf',sub_folder='over_subs/task')


# pl.open_figure(force=1)
# avg_rts = {}
# avg_rts['PP'] = []
# avg_rts['PU'] = []
# avg_rts['UP'] = []
# avg_rts['UU'] = []

# for subname in sublist:
# 	# avg_rts['PP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
# 	avg_rts['PU'].append(np.mean(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UP') * (all_rts.loc[subname]['correct']==1)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==1)])))
# 	avg_rts['UP'].append(np.mean(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PU') * (all_rts.loc[subname]['correct']==1)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==1)])))
# 	avg_rts['UU'].append(np.mean(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UU') * (all_rts.loc[subname]['correct']==1)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==1)])))


# pl.bar_plot(data = avg_rts, conditions = ['UP','PU','UU'], with_error = True, ylabel='Reaction time (% of predicted)', y_lim = [1.0, 1.2])

# # pl.bar_plot
# pl.save_figure('rt_correct_norm.pdf',sub_folder='over_subs/task')



# pl.open_figure(force=1)
# avg_rts = {}
# avg_rts['PP'] = []
# avg_rts['PU'] = []
# avg_rts['UP'] = []
# avg_rts['UU'] = []

# for subname in sublist:
# 	avg_rts['PP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
# 	avg_rts['PU'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PU') * (all_rts.loc[subname]['correct']==0)]))
# 	avg_rts['UP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UP') * (all_rts.loc[subname]['correct']==0)]))
# 	avg_rts['UU'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UU') * (all_rts.loc[subname]['correct']==0)]))

# avg_rts['PP'] = np.array(avg_rts['PP'])
# avg_rts['UP'] = np.array(avg_rts['UP']) 
# avg_rts['UU'] = np.array(avg_rts['UU'])
# avg_rts['PU'] = np.array(avg_rts['PU'])


# pl.bar_plot(data = avg_rts, conditions = ['PP','UP','PU','UU'], with_error = True, ylabel='Reaction time (s)')

# # pl.bar_plot
# pl.save_figure('rt_incorrect_raw.pdf',sub_folder='over_subs/task')

# pl.open_figure(force=1)
# avg_rts = {}
# avg_rts['PP'] = []
# avg_rts['PU'] = []
# avg_rts['UP'] = []
# avg_rts['UU'] = []

# for subname in sublist:
# 	# avg_rts['PP'].append(np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
# 	avg_rts['PU'].extend(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PU') * (all_rts.loc[subname]['correct']==0)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
# 	avg_rts['UP'].extend(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UP') * (all_rts.loc[subname]['correct']==0)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))
# 	avg_rts['UU'].extend(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='UU') * (all_rts.loc[subname]['correct']==0)] / np.median(all_rts.loc[subname]['reaction_time'][(all_rts.loc[subname]['condition']=='PP') * (all_rts.loc[subname]['correct']==0)]))


# pl.bar_plot(data = avg_rts, conditions = ['UP','PU','UU'], with_error = True, ylabel='Reaction time (% of predicted)', y_lim = [1.0, 1.2])

# # pl.bar_plot
# pl.save_figure('rt_incorrect_norm.pdf',sub_folder='over_subs/task')

# # embed()