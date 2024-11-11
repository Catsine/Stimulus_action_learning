#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:21:39 2024

@author: cate
"""
import numpy as np
import pandas as pd
import statistics

# =============================================================================
# define function to properly plot circular data

def circular_mean(lista):
    listaRad= (np.deg2rad(lista))
    mean_cos= statistics.mean(np.cos(listaRad))
    mean_sin= statistics.mean(np.sin(listaRad))
    x= np.arctan2(mean_sin,mean_cos)
    x= np.rad2deg(x)
    return x

def circular_stddev(angles):
    anglesRad= (np.deg2rad(angles))
    anglesRad= np.array(anglesRad)
    mean_angle= np.mean(np.exp(1j * anglesRad))
    stddevRad= np.sqrt(-2*np.log(np.abs(mean_angle)))
    stddev=np.rad2deg(stddevRad)
    return stddev

def circular_median(lista):
    listaRad= (np.deg2rad(lista))
    median_cos= statistics.median(np.cos(listaRad))
    median_sin= statistics.median(np.sin(listaRad))
    x= np.arctan2(median_sin,median_cos)
    x= np.rad2deg(x)
    return x

# Plot interaction plot - 2x2
def within_subjects_error_2(data,concatenated_data,dv,var_a,var_b,var_a1,var_a2,var_b1,var_b2,subjects):
    ''' Function to compute within-subject errors in repeated measures designes with 2x2 categorical factors
    data =  dataframe with dependent variable
    var_a = factor a
    var_b = factor b
    a1,a2 = levels of factor a
    b1,b2 = levels of factor b
    subjects = list of participants id '''

    grand_average  = concatenated_data[dv].mean()
    v_a1b1, v_a1b2, v_a2b1, v_a2b2 = [], [], [], []
    for i in subjects:
        subject_average = data[dv][data['id']==i].mean()
        v_a1b1.append(data[dv][(data['id']==i) & (data[var_a]==var_a1) & (data[var_b]==var_b1)].mean() - subject_average + grand_average)
        v_a1b2.append(data[dv][(data['id']==i) & (data[var_a]==var_a1) & (data[var_b]==var_b2)].mean() - subject_average + grand_average)
        v_a2b1.append(data[dv][(data['id']==i) & (data[var_a]==var_a2) & (data[var_b]==var_b1)].mean() - subject_average + grand_average)
        v_a2b2.append(data[dv][(data['id']==i) & (data[var_a]==var_a2) & (data[var_b]==var_b2)].mean() - subject_average + grand_average)

    std_v_a1b1, std_v_a1b2, std_v_a2b1, std_v_a2b2 = map(lambda x: statistics.stdev(x), [v_a1b1, v_a1b2, v_a2b1, v_a2b2])
    SE_v_a1b1,SE_v_a1b2, SE_v_a2b1, SE_v_a2b2 = map(lambda x: x/np.sqrt(len(subjects)), [std_v_a1b1, std_v_a1b2, std_v_a2b1, std_v_a2b2])   

    return SE_v_a1b1,SE_v_a1b2, SE_v_a2b1, SE_v_a2b2

# for graph with all orientations
def within_subjects_error(data,concatenated_data,error_sign,var_a,var_b,var_a1,var_a2,var_a3,var_a4,var_a5,var_a6,var_a7,var_b1,var_b2,subjects):
    
    ''' Function to compute within-subject errors in repeated measures designes with nxn categorical factors
    data =  dataframe with reaction_times column
    var_a = factor a
    var_b = factor b
    a1,a2,a3,a4,a5,a6,a7 = levels of factor a
    b1,b2 = levels of factor b
    subjects = list of participants id '''

    grand_average  = concatenated_data[error_sign].mean()
    v_a1b1, v_a1b2, v_a2b1, v_a2b2, v_a3b1, v_a3b2, v_a4b1, v_a4b2, v_a5b1, v_a5b2, v_a6b1, v_a6b2, v_a7b1, v_a7b2 = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in subjects:
        subject_average = data[error_sign][data['id']==i].mean()
        v_a1b1.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a1) & (data[var_b]==var_b1)].mean() - subject_average + grand_average)
        v_a1b2.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a1) & (data[var_b]==var_b2)].mean() - subject_average + grand_average)
        v_a2b1.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a2) & (data[var_b]==var_b1)].mean() - subject_average + grand_average)
        v_a2b2.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a2) & (data[var_b]==var_b2)].mean() - subject_average + grand_average) 
        v_a3b1.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a3) & (data[var_b]==var_b1)].mean() - subject_average + grand_average)
        v_a3b2.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a3) & (data[var_b]==var_b2)].mean() - subject_average + grand_average)
        v_a4b1.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a4) & (data[var_b]==var_b1)].mean() - subject_average + grand_average)
        v_a4b2.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a4) & (data[var_b]==var_b2)].mean() - subject_average + grand_average)
        v_a5b1.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a5) & (data[var_b]==var_b1)].mean() - subject_average + grand_average)
        v_a5b2.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a5) & (data[var_b]==var_b2)].mean() - subject_average + grand_average)
        v_a6b1.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a6) & (data[var_b]==var_b1)].mean() - subject_average + grand_average)
        v_a6b2.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a6) & (data[var_b]==var_b2)].mean() - subject_average + grand_average)
        v_a7b1.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a7) & (data[var_b]==var_b1)].mean() - subject_average + grand_average)
        v_a7b2.append(data[error_sign][(data['id']==i) & (data[var_a]==var_a7) & (data[var_b]==var_b2)].mean() - subject_average + grand_average)

    std_v_a1b1, std_v_a1b2, std_v_a2b1, std_v_a2b2,std_v_a3b1, std_v_a3b2, std_v_a4b1, std_v_a4b2, std_v_a5b1, std_v_a5b2,std_v_a6b1, std_v_a6b2,std_v_a7b1, std_v_a7b2 = map(lambda x: circular_stddev(x), [v_a1b1, v_a1b2, v_a2b1, v_a2b2,v_a3b1, v_a3b2, v_a4b1, v_a4b2, v_a5b1, v_a5b2, v_a6b1, v_a6b2, v_a7b1, v_a7b2])
    SE_v_a1b1,SE_v_a1b2, SE_v_a2b1, SE_v_a2b2,SE_v_a3b1, SE_v_a3b2, SE_v_a4b1, SE_v_a4b2, SE_v_a5b1, SE_v_a5b2,SE_v_a6b1, SE_v_a6b2,SE_v_a7b1, SE_v_a7b2 = map(lambda x: x/np.sqrt(len(subjects)), [std_v_a1b1, std_v_a1b2, std_v_a2b1, std_v_a2b2,std_v_a3b1, std_v_a3b2, std_v_a4b1, std_v_a4b2, std_v_a5b1, std_v_a5b2,std_v_a6b1, std_v_a6b2,std_v_a7b1, std_v_a7b2])   

    return SE_v_a1b1,SE_v_a1b2, SE_v_a2b1, SE_v_a2b2,SE_v_a3b1, SE_v_a3b2, SE_v_a4b1, SE_v_a4b2, SE_v_a5b1, SE_v_a5b2,SE_v_a6b1, SE_v_a6b2,SE_v_a7b1, SE_v_a7b2
