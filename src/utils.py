import argparse
import os
import pickle
import pprint
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os


import matplotlib.ticker as ticker
import pdb
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# prepare rolling sums
def get_rolling_amount(grp, freq, amt, time):
    return grp.rolling(freq, on=time, min_periods=0)[amt].sum() #min_periods=0

def get_rolling_df(df, time='RESULT', freq=7, var='covid'):
    df = df[[time, var]]
    #print('df befored fill in dates:', len(df[time].unique()))
    min_date = df[time].min()
    max_date = df[time].max()
    # fill in every date
    dates = pd.date_range(min_date, max_date)
    var_val = [0]*len(dates)
    everylevel = pd.DataFrame({time:dates, var:var_val})
    df = df.merge(everylevel, on=[time, var], how='outer')
    
    df.sort_values(by=time,inplace=True)

    #print('df after fill in dates:', len(df[time].unique()))

    
    freqd = '{}D'.format(freq)
    df.set_index(time, inplace=True)

    df = df.rolling(freqd, min_periods=0)[var].sum().reset_index()
    df = df.set_index(time)

    df = df.groupby(time)[var].max().reset_index()
    df = df.set_index(time)

   # print('after_rolling', df[(df.index=='2020-03-08')])
    df[var].fillna(0, inplace=True)
    
    df[var] = df[var]/freq
    df.sort_values(by=time,inplace=True)
    
    #print("end of daily----------------------")
    return df


def get_groupby_rolling_df(df, grpvar='Age group', freq='7D', amt="Died", time='Calendar date'):
    #print(df[(df['Calendar date']>='2020-07-01')&(df['Calendar date']<='2020-07-07') & (df['Age group']=='65-74')][[var]].sum())
    #cats = df[grpvar].cat.categories
    cats = ['0-9',
                                             '10-19',
                                             '20-29',
                                             '30-39',
                                             '40-49',
                                             '50-59',
                                             '60-69',
                                             '70-79',
                                             '80+']
    out='rolling sum'
    #print(df[(df['Calendar date']>='2020-07-01')&(df['Calendar date']<='2020-07-07') & (df['Age group']=='65-74')][['Calendar date', 'Age group', 'rolling sum']])
    df = df[[time, grpvar, amt]]
    # merge on precomputed
    
    
    min_date = df[time].min()
    max_date = df[time].max()
    dates = pd.date_range(min_date, max_date)
    
    everylevel = pd.DataFrame(list(product(dates.unique(), cats)), columns=[time, grpvar])
    everylevel[amt] = 0
    #print(everylevel[0:3])
    #df4 = df3[['Age_group','HFR']].merge(df.loc[:, df.columns != time], on=['Age_group', time], how='inner')
   # everylevel.set_index(time, inplace=True)
   # cats = df[grpvar].cat.categories
    df = df.merge(everylevel, on=[time, grpvar, amt], how='outer')
    df.sort_values(by=time,inplace=True)
    df[grpvar] = df[grpvar].astype('category')
    df[grpvar].cat.reorder_categories(cats, inplace=True)

    df[out]= df.groupby([grpvar],  as_index=False, group_keys=False) \
                               .apply(get_rolling_amount, freq, amt, time)                          
                            #columns=[time, grpvar])

    df = df.groupby([grpvar, time])[out].max().reset_index()
    #df = df.set_index(time)
    
   # print('max_outed',df[(df[time]=='2020-03-08')])

    df[out].fillna(0, inplace=True)
    df = df[[grpvar, time, out]]
    df[out] = df[out]/7
    #print('averaged',df[(df.index=='2020-03-14')])

    df = df.set_index(time)
   # print("end of df_te----------------------")
    return df


# helper functions for graphing
def graph_leg_acc(p, name):
    min_date = "2020-04-01"
    max_date = "2020-12-01"
    grey = False
    p.set_xlim(pd.Timestamp(min_date), pd.Timestamp(max_date))
    grey = False
    if grey:
        if name!='COVID-19 Cases':
            p.axvspan(pd.Timestamp(max_date)-pd.Timedelta(days=30), pd.Timestamp(max_date), color='grey', alpha=0.5, lw=0)
        else: 
            p.axvspan(pd.Timestamp(max_date)-pd.Timedelta(days=14), pd.Timestamp(max_date), color='grey', alpha=0.5, lw=0)

    p.set_title(name)


def graph_leg(p, name):
    #fig = p[0].get_figure()
   # p[0].set_xlabel("")
    min_date = "2020-04-01"
    max_date = "2020-12-1"
    p[0].set_xlim(pd.Timestamp(min_date), pd.Timestamp(max_date))
    grey = False
    if grey: 
        if name!='COVID-19 Cases':
            p[0].axvspan(pd.Timestamp(max_date)-pd.Timedelta(days=30), pd.Timestamp(max_date), color='grey', alpha=0.5, lw=0)
        else: 
            p[0].axvspan(pd.Timestamp(max_date)-pd.Timedelta(days=14), pd.Timestamp(max_date), color='grey', alpha=0.5, lw=0)

    p[0].set_title(name)


    
def remove_unkown(df, grpvar):
    df = df[df[grpvar]!='_Unknown']
    df[grpvar].cat.remove_unused_categories(inplace=True)
    return df

def test():
    print('test import')
    
    
def prepare_hfr_age(newdf, grpvar, freq, amt, time):
    min_date = "2020-04-01"
    max_date = "2020-12-1"
    df = get_groupby_rolling_df(newdf, grpvar, freq, amt, time)
    df1 = df.copy()
    df1['Died'] = df1['rolling sum']
    print("df1", df1.shape)
    # get hospitalized
    df= get_groupby_rolling_df(newdf, grpvar, freq, 'Hospitalized', time)
    df2 = df
    df2['Hospitalized'] = df2['rolling sum']

    df3 = df1
    df3['Hospitalized'] = df2['Hospitalized']
    df3[['Died', 'Hospitalized']].fillna(0, inplace=True)
    df3 = df3[(df3.index>=min_date) & (df3.index<max_date)]
    return df3
   
def get_mean_hfr_age(newdf, time):
    grpvar='Age_group'
    freq='7D'
    amt="Died"
    time=time
    df3 = prepare_hfr_age(newdf, grpvar, freq, amt, time)
    
    df3 = df3.groupby('Age_group')[['Died', 'Hospitalized']].sum().reset_index()
    df3['HFR'] = df3['Died']/df3['Hospitalized']
    #print(df3)
    #df3.to_csv('new_hfr_country.csv')
    return df3


def prepare_cfr_age(newdf, grpvar, freq, amt, time):
    min_date = "2020-04-01"
    max_date = "2020-12-1"
    df = get_groupby_rolling_df(newdf, grpvar, freq, 'Died', time)
    df1 = df.copy()
    df1['Died'] = df1['rolling sum']
    # all hosp
    df = get_groupby_rolling_df(newdf, grpvar, freq, 'Hospitalized', time)
    df2 = df.copy()
    df2['Hospitalized'] = df2['rolling sum']
    #all
    df = get_groupby_rolling_df(newdf, grpvar, freq, 'all', time)
    df['all'] = df['rolling sum']


    df = df[[grpvar, 'all']].merge(df1[[grpvar, 'Died']], on=[time, grpvar], how='outer')
    df = df[[grpvar, 'all', 'Died']].merge(df2[[grpvar, 'Hospitalized']], on=[time, grpvar], how='outer')
    df[['Died', 'Hospitalized']].fillna(0, inplace=True)
    
    df = df[(df.index>=min_date) & (df.index<max_date)]
    return df

def get_mean_cfr_age(newdf, time):
    grpvar='Age_group'
    freq='7D'
    amt='all'
    time=time
    df = prepare_cfr_age(newdf, grpvar, freq, amt, time)
    df = df.groupby('Age_group')[['Died', 'all']].sum().reset_index()
    df['CFR'] = df['Died']/df['all']
    #print(df)
    #df.to_csv('new_cfr_countr.csv')
    return df   
    
def get_age_ratio(amt, newdf, time):
    #newdf = florida.copy()
    #amt='Hospitalized'
    freq='7D'
    grpvar='Age_group'
    df = get_groupby_rolling_df(newdf, grpvar, freq, amt, time)
    df[amt] = df['rolling sum']
    daily = get_rolling_df(newdf, time=time, freq=7, var=amt)
    daily[amt].fillna(0, inplace=True)
    daily.rename(columns={amt:'daily_total'}, inplace=True)
    merged = df.merge(daily, left_index=True, right_index=True)
    merged['ratio'] = merged[amt] / merged['daily_total']
    age_ratio = merged[['ratio', grpvar]]
    return age_ratio
    
def get_from_mean(newdf, amt, ax, time):
    min_date = "2020-04-01"
    max_date = "2020-12-01"
    grpvar = 'Age_group'
    age_ratio = get_age_ratio(amt, newdf, time)
    if amt=='Hospitalized': # HFR
        hfr_age = get_mean_hfr_age(newdf[newdf[amt] ==1].copy(), time) 
        age_ratio['mean_HFR'] = age_ratio.apply(lambda x: hfr_age[hfr_age[grpvar]==x[grpvar]]['HFR'].item(), axis=1)
    else: # CFR
        hfr_age = get_mean_cfr_age(newdf[newdf[amt] ==1].copy(), time) 
        age_ratio['mean_HFR'] = age_ratio.apply(lambda x: hfr_age[hfr_age[grpvar]==x[grpvar]]['CFR'].item(), axis=1)
    
    age_ratio['mHFR'] = age_ratio['ratio'] * age_ratio['mean_HFR']
    p=age_ratio.groupby(time)['mHFR'].sum().plot(ax=ax)
    p.set_xlim(pd.Timestamp(min_date), pd.Timestamp(max_date))

def graph_leg_save_bar(p, name, thre):
    grey = False
    if grey:
        p.axvspan(p.get_xlim()[1]*thre, p.get_xlim()[1], color='grey', alpha=0.5, lw=0)
    #p[0].set_xlim(pd.Timestamp(min_date), pd.Timestamp(max_date))
    p.set_title(name)
    
    
def get_bar_chart(amt, newdf, time, name, ax, it, grpvar='Age_group'):
    min_date = "2020-04-01"
    max_date = "2020-12-01"
    #grpvar = 'state'
    if grpvar == 'Age_group':
        newdf = remove_unkown(newdf, grpvar)
    freq = '7D'
    df = get_groupby_rolling_df(newdf, grpvar, freq, amt, time)
    df[amt] = df['rolling sum']
    daily = get_rolling_df(newdf, time=time, freq=7, var=amt)
    daily[amt].fillna(0, inplace=True)
    daily.rename(columns={amt:'died_daily'}, inplace=True)

    merged = df.merge(daily, left_index=True, right_index=True)

    # merged

    merged['died_ratio'] = merged[amt] / merged['died_daily']
    
    m = merged[(merged.index >=min_date) & (merged.index<=max_date)]
    if it==1:
        m.to_csv('case_ratio_country.csv')
    #m.fillna(0, inplace=True)
    ylb = 'Ratio' if it==0 else ''
    xlb = 'CDC Report Date' if it==1 else ''
    p = m.set_index(grpvar,append=True)['died_ratio'].unstack().plot(ylabel=ylb, xlabel=xlb, stacked=True, ax=ax, kind='bar', legend=False)


    # Make most of the ticklabels empty so the labels don't get too crowded
    ticklabels = ['']*len(m.index.unique())
    # Every 4th ticklable shows the month and day
    ticklabels[::1] = ['{} {} {}'.format(item.year, item.month, item.day)for item in m.index.unique()[::1]]
    # Every 12th ticklabel includes the year
    #ticklabels[::12] = [item.strftime('%b %d\n%Y') for item in m.index[::12]]
    t1 = []
    thre = 0
    for i, item in enumerate(ticklabels):
        #print(item)
        y, m, d = item.split(" ")
        blackout = 14 if name == 'COVID-19 Cases' else 30
        max_item = (newdf[time].max()- pd.Timedelta(days=blackout))
        max_y, max_m, max_d = max_item.year, max_item.month, max_item.day
        if m==str(max_m) and d==str(max_d):
            thre = i+1
        if d!='1':
            t1.append("")
        else:
            t1.append("2020-{}".format(m))

    p.xaxis.set_major_formatter(ticker.FixedFormatter(t1))
    #p.set_xlim(pd.Timestamp('2020-03-11'), pd.Timestamp('2020-08-02')) 

    plt.gcf().autofmt_xdate()
    graph_leg_save_bar(p, name, thre/len(t1))   