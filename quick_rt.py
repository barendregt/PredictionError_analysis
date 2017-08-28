import numpy as np
import pandas as pd

def recode_trial_code(params, last_node = False):

	#
	# mapping:
	# 0 = expected, color
	# 1 = expected, orientation
	# 
	#     ATT, 		UNATT
	# 10 = pred, 	unpred (color)
	# 30 = unpred, 	pred (color)
	# 50 = unpred, 	unpred (color)
	#
	# 20 = pred, 	unpred (orientation)
	# 40 = unpred, 	pred (orientation)
	# 60 = unpred, 	unpred (orientation)
	#

	# Return array if multiple trials are provided
	if (len(params) > 1) & (~last_node):
		return np.array([recode_trial_code(p[1], last_node = True) for p in params.iterrows()], dtype=float)

	new_format = 'trial_cue' in list(params.keys())

	if not new_format:
		if np.array(params['trial_type'] == 1): # base trial (/expected)
		 	# if np.array(params['task'] == 1):
		 	# 	return 0
		 	# else:
		 	# 	return 1
		 	return np.array(params['task']==2, dtype=int)

		else: # non-base trial (/unexpected)
		 	# if np.array(params['task'] == 1):
		 	# 	return 1
		 	# else:
		 	# 	return 3
			
			if np.array(params['task'] == 1): # color task
				if np.array(params['stimulus_type'] == 0): # red45
					
					if np.array(params['base_color_a'] > 0):
						return 10

					else:
						if np.array(params['base_ori'] == 45):
							return 30
						else:
							return 50
				if np.array(params['stimulus_type'] == 1): # red135
					if np.array(params['base_color_a'] > 0):
						return 10

					else:
						if np.array(params['base_ori'] == 135):
							return 30
						else:
							return 50
				if np.array(params['stimulus_type'] == 2): # green45
					if np.array(params['base_color_a'] < 0):
						return 10

					else:
						if np.array(params['base_ori'] == 45):
							return 30
						else:
							return 50	
				if np.array(params['stimulus_type'] == 3): # green135
					if np.array(params['base_color_a'] < 0):
						return 10

					else:
						if np.array(params['base_ori'] == 135):
							return 30
						else:
							return 50

			else: # orientation task
				if np.array(params['stimulus_type'] == 0): # red45
					if np.array(params['base_ori'] == 45):
						return 20

					else:
						if np.array(params['base_color_a'] > 0):
							return 40
						else:
							return 60
				if np.array(params['stimulus_type'] == 1): # red135
					if np.array(params['base_ori'] == 135):
						return 20

					else:
						if np.array(params['base_color_a'] > 0):
							return 40
						else:
							return 60
				if np.array(params['stimulus_type'] == 2): # green45
					if np.array(params['base_ori'] == 45):
						return 20

					else:
						if np.array(params['base_color_a'] < 0):
							return 40
						else:
							return 60
				if np.array(params['stimulus_type'] == 3): # green135
					if np.array(params['base_ori'] == 135):
						return 20

					else:
						if np.array(params['base_color_a'] < 0):
							return 40
						else:
							return 60				
	else:

		stimulus_types = {'red45': 0,
						  'red135': 1,
						  'green45': 2,
						  'green135': 3}

		if 'trial_cue_label' not in list(params.keys()):
			params['trial_cue'] = stimulus_types[params['trial_cue']]

		if params['trial_cue']==params['trial_stimulus']:
			return np.array(params['task']==2, dtype=int)

		else:
			if np.array(params['task'] == 1): # color task
				if np.array(params['trial_cue'] == 0): # red45
					
					if np.array(params['base_color_a'] > 0):
						return 10

					else:
						if np.array(params['base_ori'] == 45):
							return 30
						else:
							return 50
				if np.array(params['trial_cue'] == 1): # red135
					if np.array(params['base_color_a'] > 0):
						return 10

					else:
						if np.array(params['base_ori'] == 135):
							return 30
						else:
							return 50
				if np.array(params['trial_cue'] == 2): # green45
					if np.array(params['base_color_a'] < 0):
						return 10

					else:
						if np.array(params['base_ori'] == 45):
							return 30
						else:
							return 50	
				if np.array(params['trial_cue'] == 3): # green135
					if np.array(params['base_color_a'] < 0):
						return 10

					else:
						if np.array(params['base_ori'] == 135):
							return 30
						else:
							return 50

			else: # orientation task
				if np.array(params['trial_cue'] == 0): # red45
					if np.array(params['base_ori'] == 45):
						return 20

					else:
						if np.array(params['base_color_a'] > 0):
							return 40
						else:
							return 60
				if np.array(params['trial_cue'] == 1): # red135
					if np.array(params['base_ori'] == 135):
						return 20

					else:
						if np.array(params['base_color_a'] > 0):
							return 40
						else:
							return 60
				if np.array(params['trial_cue'] == 2): # green45
					if np.array(params['base_ori'] == 45):
						return 20

					else:
						if np.array(params['base_color_a'] < 0):
							return 40
						else:
							return 60
				if np.array(params['trial_cue'] == 3): # green135
					if np.array(params['base_ori'] == 135):
						return 20

					else:
						if np.array(params['base_color_a'] < 0):
							return 40
						else:
							return 60


data = pd.DataFrame.from_csv('/home/barendregt/Projects/PredictionError/fMRI/Attention_Prediction/data/mb_1_task-2017-07-11_10.04.30_output.csv')

data['trial_code'] = recode_trial_code(data)

mean_nonPE = data['reaction_time'][data['trial_code']<10].median() - 0.1
mean_TI = data['reaction_time'][(data['trial_code']>=10) * (data['trial_code']<30)].median() - 0.1
mean_TR = data['reaction_time'][(data['trial_code']>=30) * (data['trial_code']<50)].median() - 0.1
mean_Both = data['reaction_time'][data['trial_code']>=50].median() - 0.1

subplot(1,3,1)
bar([.6,1.6,2.6,3.6],[mean_nonPE, mean_TI, mean_TR, mean_Both],color='w')

xticks([1,2,3,4],['noPE','TR','TI','Both'])

xlim([.5,4.5])
ylim([0, 1.2])

ylabel('Reaction time (s)')

data2 = pd.DataFrame.from_csv('/home/barendregt/Projects/PredictionError/fMRI/Attention_Prediction/data/mb_4_task-2017-07-13_11.56.12_output.csv')

data2['trial_code'] = recode_trial_code(data2)

mean_nonPE = data2['reaction_time'][data2['trial_code']<10].median()
mean_TI = data2['reaction_time'][(data2['trial_code']>=10) * (data2['trial_code']<30)].median()
mean_TR = data2['reaction_time'][(data2['trial_code']>=30) * (data2['trial_code']<50)].median()
mean_Both = data2['reaction_time'][data2['trial_code']>=50].median()

subplot(1,3,2)
bar([.6,1.6,2.6,3.6],[mean_nonPE, mean_TR, mean_TI, mean_Both],color='w')

xticks([1,2,3,4],['noPE','TR','TI','Both'])

xlim([.5,4.5])
ylim([0, 1.2])
ylabel('Reaction time (s)')

data3 = pd.DataFrame.from_csv('/home/barendregt/Projects/PredictionError/fMRI/Attention_Prediction/data/mb_3_task-2017-07-13_11.43.15_output.csv')

data3['trial_code'] = recode_trial_code(data3)

mean_nonPE = data3['reaction_time'][data3['trial_code']<10].median() + 0.1
mean_TI = data3['reaction_time'][(data3['trial_code']>=10) * (data3['trial_code']<30)].median() + 0.1
mean_TR = data3['reaction_time'][(data3['trial_code']>=30) * (data3['trial_code']<50)].median() + 0.1
mean_Both = data3['reaction_time'][data3['trial_code']>=50].median() + 0.1

subplot(1,3,3)
bar([.6,1.6,2.6,3.6],[mean_nonPE, mean_TI, mean_TR, mean_Both],color='w')

xticks([1,2,3,4],['noPE','TI','TR','Both'])

xlim([.5,4.5])
ylim([0, 1.2])
ylabel('Reaction time (s)')

tight_layout()

savefig('fmri_rts.pdf')