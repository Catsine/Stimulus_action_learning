#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:05:26 2024

@author: cate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import biased_memory_toolbox as bmt
from itertools import repeat
from functions import *

### import and clean files
dfs = {}
dfs_task_2 = {}
subjects =  [['3','33'],['4','44'],['5','55'],['6','66']]

for subj in subjects:
    
    df1 = pd.read_csv(r'//Users//cate//Documents//Postdoc//1st_project//subject-{}.csv'.format(subj[0])) 
    if subj[1] =='33' or subj[1]=='44' or subj[1]=='55' or subj[1]=='66':
        df2 = pd.read_excel(r'//Users//cate//Documents//Postdoc//1st_project//subject-{}.xlsx'.format(subj[1])) 
    elif subj[1]=='11' or subj[1]=='22':
        df2 = pd.read_csv(r'//Users//cate//Documents//Postdoc//1st_project//subject-{}.csv'.format(subj[1])) 
    
    #1) remove practice trials 
    # first session
    initial_practice_1 = list(range(0,24))
    n=32
    m=3
    later_practice_1 = [j for i in range(1,585, n+m) for j in range(i, m+i)]
    practice_1 = initial_practice_1 + later_practice_1   
    
    # second session
    initial_practice_2 = list(range(0,8))
    n=32
    m=3
    later_practice_2 = [j for i in range(1,428, n+m) for j in range(i, m+i)]
    exclude_last_task = list(range(428,len(df2)))
    practice_2 = initial_practice_2 + later_practice_2 + exclude_last_task
    
    # drop practice trials
    df1 = df1.drop(df1.index[[practice_1]])
    df3 = df2[428:]
    df2 = df2.drop(df2.index[[practice_2]])
    df = pd.concat([df1,df2])
    df['reset_index'] = list(range(len(df)))
    df = df.set_index('reset_index')
    df['subject_nr'] = int(subj[0])
    df3['subject_nr'] = int(subj[0])
    #2) drop all trials with mistaken or missed movement (both first and second) and lifted finger
    df = df[((df['correct_action']=='1') | (df['correct_action']==1)) & ((df['correct_action_sec']==1)|(df['correct_action_sec']=="1")) & (df['finger_was_lifted_mem_delay']==0)]
    df = df[(df['repOri_opensesame'] != 0.111111111111) | (df['repOri_sec_opensesame'] != 0.111111111111) | (df['repOri_opensesame'] != 500) | (df['repOri_sec_opensesame'] != 500)]
    
    df['reset_index'] = list(range(len(df)))
    df = df.set_index('reset_index')
    
    #3) compute error for the first probe (first report)
    error = []
    prova = []
    for i in range(len(df)):
        if df['repOri_opensesame'][i]-df['origOri_opensesame'][i]>=180:
            error.append(df['repOri_opensesame'][i]-df['origOri_opensesame'][i]-360) #negative error CCW
        elif df['repOri_opensesame'][i]-df['origOri_opensesame'][i]<-180:
                error.append(df['repOri_opensesame'][i]-df['origOri_opensesame'][i]+360)#positive error CW
        else:
            error.append(df['repOri_opensesame'][i]-df['origOri_opensesame'][i])
        prova.append(df['repOri_opensesame'][i]-df['origOri_opensesame'][i])
    df['error_']=error  
    ### compute error for the second probe (second report)
    error_sec = []
    for i in range(len(df)):
        if df['repOri_sec_opensesame'][i]-df['origOri_sec_opensesame'][i]>=180:
            error_sec.append(df['repOri_sec_opensesame'][i]-df['origOri_sec_opensesame'][i]-360)#negative error CCW
        elif df['repOri_sec_opensesame'][i]-df['origOri_sec_opensesame'][i]<=-180:
            error_sec.append(df['repOri_sec_opensesame'][i]-df['origOri_sec_opensesame'][i]+360)#positive error CW
        else:
            error_sec.append(df['repOri_sec_opensesame'][i]-df['origOri_sec_opensesame'][i])    
    df['error_sec']=error_sec

    #4) compute the sign of the error for the first probe
    sign_error = []
    sign_error_2 = []
    for i in range(len(df)):
        if np.abs(df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i])<=180:
            if df['error_'][i]>=0 and df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]>=0:#attraction
                sign_error.append(-df['error_'][i])
            elif df['error_'][i]>=0 and df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]<=0:#repulsion
                sign_error.append(df['error_'][i])
            elif df['error_'][i]<=0 and df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]<=0:#attraction
                sign_error.append(df['error_'][i])
            elif df['error_'][i]<=0 and df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]>=0:#repulsion
                sign_error.append(-df['error_'][i])
        elif df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]>180:
            if df['error_'][i]>=0 and (df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]-360)<=0:#repulstion
                sign_error.append(df['error_'][i])#it was plus
            elif df['error_'][i]<=0 and (df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]-360)<=0:#attraction
                sign_error.append(df['error_'][i])
        elif df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]<-180:
            if df['error_'][i]>=0 and (df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]+360)>=0:#attraction
                sign_error.append(-df['error_'][i])
            elif df['error_'][i]<=0 and (df['origOri_sec_opensesame'][i]-df['origOri_opensesame'][i]+360)>=0:#repulsion
                sign_error.append(-df['error_'][i])
    ### compute the sign of the error for the second probe    
    for i in range(len(df)):
        if np.abs(df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i])<=180:
            if df['error_sec'][i]>=0 and df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]>0:#attraction
                sign_error_2.append(-df['error_sec'][i])
            elif df['error_sec'][i]>=0 and df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]<0:#repulsion
                sign_error_2.append(df['error_sec'][i])
            elif df['error_sec'][i]<=0 and df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]<0:#attraction
                sign_error_2.append(df['error_sec'][i])
            elif df['error_sec'][i]<=0 and df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]>0:#repulsion
                sign_error_2.append(-df['error_sec'][i])
        elif df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]>180:
            if df['error_sec'][i]>=0 and (df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]-360)<0:#repulstion
                sign_error_2.append(df['error_sec'][i])
            elif df['error_sec'][i]<=0 and (df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]-360)<0:#attraction
                sign_error_2.append(df['error_sec'][i])
        elif df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]<-180:
            if df['error_sec'][i]>=0 and (df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]+360)>0:#attraction
                sign_error_2.append(-df['error_sec'][i])
            elif df['error_sec'][i]<=0 and (df['origOri_opensesame'][i]-df['origOri_sec_opensesame'][i]+360)>0:#repulsion
                sign_error_2.append(-df['error_sec'][i])                   
    ### collapse positive and negative angolar differences together  


    df['diffs']= np.abs(df['orienta_red']-df['orienta_blue'])
    df['error_sign'] = sign_error
    df['error_2_sign'] = sign_error_2
    df['error_sign_rads'] = np.radians(sign_error)
    data = {'{}'.format(subj[0]): df}
    data2 = {'{}'.format(subj[0]): df3}
    dfs_task_2.update(data2)
    dfs.update(data)
    
dataset = pd.concat(dfs.values())
dataset2 = pd.concat(dfs_task_2.values())
### drop outlier trials in terms of the error (considering also the sign)
# the data gets cleaned separately per subject per task per action condition per angular differences
learning = ['random','fixed']
condition = ['different','same']

data_subject = []
mu1_subject = []
std1_subject = []
mu2_subject = []
std2_subject = []

# Check how many datapoints remain per participant:
subjects_ = [3,4,5,6]#,4,5,6]

len_dt = []
# for each subject
for sj in subjects_:
    
  data_learning = []
  mu1_learning = []
  std1_learning = []
  mu2_learning = []
  std2_learning = []
  
  # for each learning task
  for t in learning:
      
      data_condition = []
      mu1_condition = []
      std1_condition = []
      mu2_condition = []
      std2_condition = []

      # for each action condition
      for n in condition:

          dt1 =  dataset['error_sign'][(dataset['subject_nr']==sj) & (dataset['condition']==n) & (dataset['pairs_types_first']==t)]
          mean1 = circular_mean(dt1.tolist())
          std1 = circular_stddev(dt1.tolist())
          dt2 =  dataset['error_2_sign'][(dataset['subject_nr']==sj) & (dataset['condition']==n) & (dataset['pairs_types_first']==t)]
          mean2 = circular_mean(dt2.tolist())
          std2 = circular_stddev(dt2.tolist())
    
          data_ = dataset[(dataset['subject_nr']==sj) & (dataset['condition']==n) & (dataset['pairs_types_first']==t)]
              
          outliers1 = [x for x in data_['error_sign'].to_list() if  np.abs(x-mean1)>2.5 * std1]
          outliers2 = [x for x in data_['error_2_sign'].to_list() if  np.abs(x-mean2)>2.5 * std2]
              
          dt_no_outliers1 = data_[(~data_['error_sign'].isin(outliers1))] 
          dt_no_outliers = dt_no_outliers1[(~dt_no_outliers1['error_2_sign'].isin(outliers2))]
              
          mu1 = circular_mean(dt_no_outliers['error_sign'].tolist())
          mu2 = circular_mean(dt_no_outliers['error_2_sign'].tolist())
          mean3 = circular_mean(dt_no_outliers['error_'].abs().tolist())
          mean4 = circular_mean(dt_no_outliers['error_sec'].abs().tolist())              

          data_condition.append(dt_no_outliers)
              
          mu1_condition.append(mu1)
          std1_condition.append(circular_stddev(dt_no_outliers['error_sign'].tolist()))
          mu2_condition.append(mu2)
          std2_condition.append(circular_stddev(dt_no_outliers['error_2_sign'].tolist()))       
          
      ta = pd.concat(data_condition,ignore_index = True)
         
      mu1_learning.append(mu1_condition)
      std1_learning.append(std1_condition)
      mu2_learning.append(mu2_condition)
      std2_learning.append(std2_condition)
      data_learning.append(ta)
      
  s = pd.concat(data_learning,ignore_index = True)
  
  len_dt.append(len(s))
  data_subject.append(s)
  std1_subject.append(std1_learning)
  mu1_subject.append(mu1_learning)
  std2_subject.append(std2_learning)
  mu2_subject.append(mu2_learning)
 
df_clean = pd.concat(data_subject,ignore_index=True)

# flatten data to create dataframe for the plotting
m1 = [item for items in mu1_subject for item in items]
m1_ = [item for items in m1 for item in items]

m2 = [item for items in mu2_subject for item in items]
m2_ = [item for items in m2 for item in items]

std1 = [item for items in std1_subject for item in items]
std1_ = [item for items in std1 for item in items]

std2 = [item for items in std2_subject for item in items]
std2_ = [item for items in std2 for item in items]

# create dataframe for plotting
condition_ = [['different action']+['same action']]*2
condition = [item for items in condition_ for item in items]*len(subjects_)
learning_ = [['random']*2+['fixed']*2]*len(subjects_)
learning = [item for items in learning_ for item in items]
subjects = list(np.repeat(subjects_,4))

plot = {'mu1':m1_,'mu2':m2_,'std1':std1_,'std2':std2_,'id':subjects,'condition':condition,'learning':learning}
data_plot = pd.DataFrame(plot)

palette = ["orange","teal"]
fig,ax1 = plt.subplots(figsize = (15,30))
ax1 = sns.pointplot(x = "learning", y = "mu1", hue='condition',data=data_plot, 
                  estimator=circular_mean, orient="v",palette=sns.color_palette(palette),ax=ax1,alpha=0.9,errorbar=('ci', 95))

ax1.axhline(y = 0, color = 'darkgray', linestyle = '--')

###################################################################################################
# estimate accuracy for the second task
correct_data_actions = dataset2[((dataset2['first_action']==dataset2['gesture']) & (dataset2['second_action']==dataset2['gesture_sec'])) & (dataset2['pairs_types_first']=='fixed')]
tot = len(dataset2[(dataset2['pairs_types_first']=='fixed')])

score_actions = len(correct_data_actions)/tot*100
chance = 0.25*100

print('guessed: {}'.format(score_actions),"chance: {}".format(chance))
###################################################################################################
# estimate accuracy for the second task
correct_data_actions = dataset2[((dataset2['first_action']==dataset2['gesture']) & (dataset2['second_action']==dataset2['gesture_sec'])) & (dataset2['pairs_types_first']=='random')]
tot = len(dataset2[(dataset2['pairs_types_first']=='random')])

score_actions_2 = len(correct_data_actions)/tot*100
chance = 0.25*100

print('guessed: {}'.format(score_actions_2),"chance: {}".format(chance))

plot = {'proportion correct': [score_actions,score_actions_2],'id':3,'condition':['fixed','random']}
data_plot = pd.DataFrame(plot)
fig,ax1 = plt.subplots(figsize = (15,30))
ax1 = sns.barplot(x = "condition", y = "proportion correct", data=data_plot, 
                  estimator=circular_mean, orient="v",palette=sns.color_palette(palette),ax=ax1,alpha=0.9,errorbar=('ci', 95))

ax1.axhline(y = 25, color = 'darkgray', linestyle = '--')
fig.savefig('awareness_tot.svg', format='svg', dpi=1200) 

fig,ax1 = plt.subplots(figsize = (15,30))
sns.pointplot(x = "pairs_types_first", y = "error_2_sign", hue='condition',data=df_clean, 
                  estimator=circular_mean, orient="v",palette=sns.color_palette(palette),alpha=0.9,errorbar=('ci', 95),ax=ax1)
ax1.axhline(y = 0, color = 'darkgray', linestyle = '--')
fig.savefig('sub_6_tot.svg', format='svg', dpi=1200) 
#df_clean['subject_nr']==5]
