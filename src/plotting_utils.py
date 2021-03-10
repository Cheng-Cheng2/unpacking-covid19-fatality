import argparse
import os
import pickle
import pprint
import warnings
import numpy as np
import pandas as pd
import glob
import os
from utils import *
import plotly.express as px
pd.options.plotting.backend = "plotly"


def get_datafiles_ready():
    min_date = '2020-03-26'
    max_date = "2021-02-01"
    CLEANDIR = "../../cleaned_data"
    florida = pd.read_csv(os.path.join(CLEANDIR, 'florida_2021-03-07-15-35-01.csv'),index_col=False, parse_dates=['Case_', "ChartDate"], dtype={'Age_group':'category'})
    
    print("Finished loading Florida data.")
    cdc = pd.read_csv(os.path.join(CLEANDIR, "cdc_02282021_total.csv"),index_col=False, parse_dates=['cdc_case_earliest_dt'],  dtype={'Age_group':'category'})  
    print("Finished loading CDC data.")
    florida = florida[(florida['Case_']>=min_date) & (florida['Case_']<=max_date)]
    cdc = cdc[(cdc['cdc_case_earliest_dt']>= min_date) & (cdc['cdc_case_earliest_dt']<= max_date)]
    return florida, cdc

def florida_case_hosp_death(florida, gender='All', time='Case_', race='All', state='All'):
    florida['all'] = 1
    vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
    lbs = ['COVID-19 Cases','COVID-19 Hospitalizations', 'COVID-19 Deaths']
    vnames = ['Cases', 'Hospitalizations', 'Deaths']
    grpvar='Age_group'
    freq='7D'
    #time='Case_'
    
    if gender != 'All':
        newdf = florida[florida['Gender']==gender]
    else:
        newdf = florida



    ## only for national
    if time != 'Case_':
        if race!= 'All':
            newdf = newdf[newdf['race_ethnicity_combined']==race]
        if state != 'All':
            newdf = newdf[newdf['res_state']==state]
    

    figs = []
    for i, v in enumerate(vrs):
        amt=v
        df = get_groupby_rolling_df(newdf, grpvar, freq, amt, time)
        ylb = 'Count' #if i==0 else ''
        xlb = 'Positive Confirmed Date' if time=='Case_' else 'CDC Report Date'
        df = df[df[grpvar]!='_Unknown']
        df[grpvar].cat.remove_unused_categories(inplace=True)

        df.groupby(grpvar)['rolling sum'].plot()#(ylabel=ylb, xlabel=xlb, ax=ax[i])
        df = df.reset_index()
        df = df.rename(columns={time:"Date", 'rolling sum':vnames[i], 'Age_group':'Age Group'})
        fig = px.line(df, x='Date', y=vnames[i], color='Age Group')

        showlegend = True #if i==2 else False
        fig.update_layout(xaxis_title = xlb,
                        yaxis_title = ylb,
                        legend_title = "Age Group", 
                        yaxis={'categoryorder': 'total ascending'},
                        legend={'traceorder': 'reversed', 'font':{'size':10}},
                        font = dict(size=16),
                        showlegend = showlegend,
                        margin={'l':0, 'r':0, 't':0, 'b':0},
                        width = 450,
                        height=300
                        )
        figs.append(fig)
    return figs


def florida_case_hosp_death_agg(florida, gender='All', time='Case_', race='All', state='All'):
    florida['all'] = 1
    vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
    lbs = ['COVID-19 Cases','COVID-19 Hospitalizations', 'COVID-19 Deaths']
    vnames = ['Cases', 'Hospitalizations', 'Deaths']
    grpvar='Age_group'
    freq='7D'
    #time='Case_'
    
    if gender != 'All':
        newdf = florida[florida['Gender']==gender]
    else:
        newdf = florida



    ## only for national
    if time != 'Case_':
        if race!= 'All':
            newdf = newdf[newdf['race_ethnicity_combined']==race]
        if state != 'All':
            newdf = newdf[newdf['res_state']==state]
    

    figs = []
    for i, v in enumerate(vrs):
        amt=v
        df = get_groupby_rolling_df(newdf, grpvar, freq, amt, time)
        ylb = 'Count' #if i==0 else ''
        xlb = 'Positive Confirmed Date' if time=='Case_' else 'CDC Report Date'
        #df = df[df[grpvar]!='_Unknown']
        #df[grpvar].cat.remove_unused_categories(inplace=True)

        #df.groupby(grpvar)['rolling sum'].plot()#(ylabel=ylb, xlabel=xlb, ax=ax[i])
        df = get_rolling_df(newdf, time, freq=7, var=amt)
        df = df.reset_index()
        df = df.rename(columns={time:"Date", amt:vnames[i]})
        fig = px.line(df, x='Date', y=vnames[i])

        showlegend = True #if i==2 else False
        fig.update_layout(xaxis_title = xlb,
                        yaxis_title = ylb,
                        legend_title = "Age Group", 
                        yaxis={'categoryorder': 'total ascending'},
                        legend={'traceorder': 'reversed', 'font':{'size':10}},
                        font = dict(size=16),
                        showlegend = showlegend,
                        margin={'l':0, 'r':0, 't':0, 'b':0},
                        width = 450,
                        height=300
                        )
        figs.append(fig)
    return figs

def age_distribution_plots(florida, gender='All', time='Case_', race='All', state='All'):

    florida['all'] = 1
    vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
    lbs = ['COVID-19 Cases','Hospitalizations', 'Deaths']
    vnames = ['Cases', 'Hospitalizations', 'Deaths']
    grpvar='Age_group'
    freq='7D'
    if gender != 'All':
        newdf = florida[florida['Gender']==gender]
    else:
        newdf = florida

    if time != 'Case_':
        if race!= 'All':
            newdf = newdf[newdf['race_ethnicity_combined']==race]
        if state != 'All':
            newdf = newdf[newdf['res_state']==state]

    figs = []
    for i, v in enumerate(vrs):
        amt = v
        newdf = florida.copy()
        min_date = "2020-04-01"
        max_date = "2021-02-01"
        ylb = 'Ratio' 
        xlb = 'Positive Confirmed Date' if time=='Case_' else 'CDC Report Date'
        if grpvar == 'Age_group':
            newdf = remove_unkown(newdf, grpvar)
        freq = '7D'
        df = get_groupby_rolling_df(newdf, grpvar, freq, amt, time)
        df[amt] = df['rolling sum']
        daily = get_rolling_df(newdf, time=time, freq=7, var=amt)
        daily[amt].fillna(0, inplace=True)
        daily.rename(columns={amt:'died_daily'}, inplace=True)

        merged = df.merge(daily, left_index=True, right_index=True)
        merged['died_ratio'] = merged[amt] / merged['died_daily']
        
        m = merged[(merged.index >=min_date) & (merged.index<=max_date)]
        m = m.reset_index()
        print(m.columns)
        m = m.rename(columns={time:"Date", 'died_ratio':vnames[i], 'Age_group':'Age Group'})
        
        fig = px.bar(m, x='Date', y=vnames[i], color="Age Group", barmode = 'stack')


        fig.update_layout(xaxis_title = xlb,
                        yaxis_title = ylb,
                        legend_title = "Age Group", 
                        yaxis={'categoryorder': 'total ascending'},
                        legend={'traceorder': 'reversed', 'font':{'size':12}},
                        font = dict(size=16)
                        )
        figs.append(fig)
    return figs