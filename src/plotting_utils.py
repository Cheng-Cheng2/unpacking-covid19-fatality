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
from utils import *


def get_florida_data_ready():
    DATA_DIR = "../../../florida/data/"
    time = 'Case_'
    #print(time)
    florida= pd.read_csv(os.path.join(
            DATA_DIR, 'florida_2021-01-05-15-35-01.csv'),index_col=False, parse_dates=[time, "ChartDate"])
    florida["Age_group"] = florida["Age_group"].astype('category')
    print(florida['Hospitalized'].value_counts())
    print(florida['Hospitalized'].unique()) # 'YES'=1, else = 0
    print(florida['Died'].unique()) # 'Yes'=1, nan = 0

    florida['Case'].value_counts()
    florida['Died'].fillna(0, inplace=True)
    florida.loc[florida['Died']=='Yes', 'Died'] =1 
    florida.loc[florida['Died']=='Recent', 'Died'] =1 

    florida.loc[florida['Hospitalized']!='YES', 'Hospitalized'] = 0
    florida.loc[florida['Hospitalized']=='YES', 'Hospitalized'] = 1
    #timeh = 'ChartDate' # hospitalized time
    florida.sort_values(by=time, inplace=True)
    florida[time] = florida[time].dt.date
    florida[time] = florida[time].astype('datetime64[ns]')

    # florida.loc
    order_cat = ['0-4 years',
                                                '5-14 years',
                                                '15-24 years',
                                                '25-34 years',
                                                '35-44 years',
                                                '45-54 years',
                                                '55-64 years',
                                                '65-74 years',
                                                '75-84 years',
                                                '85+ years',
                                                'Unknown']#[::-1]
    florida['Age_group'].cat.reorder_categories(order_cat, inplace=True)
    florida['Age_group'].cat.rename_categories({'Unknown':'_'+'Unknown'}, inplace=True)
    florida['Age_group'].cat.rename_categories({'0-4 years':'0-4',
                                                '5-14 years': '5-14',
                                                '15-24 years':'15-24',
                                                '25-34 years': '25-34',
                                                '35-44 years':'35-44',
                                                '45-54 years':'45-54',
                                                '55-64 years':'55-64',
                                                '65-74 years':'65-74',
                                                '75-84 years':'75-84',
                                                '85+ years':'85+'}, inplace=True)

    florida['Age_group'].cat.categories
    #max_date = florida[time].max()
    #min_date = florida[time].min()
    #print('max date:', max_date, "min date", min_date)

    min_date = pd.Timestamp('2020-04-01')
    max_date = pd.Timestamp('2020-11-1')

    florida.rename(columns={'Age_group':'Old_age_group'}, inplace=True)
    florida.columns
    florida['Age_group'] = florida['Age'].apply(lambda x: '0-9' if x < 10 
                                                            else ('10-19' if x < 20 
                                                                    else('20-29'if x < 30 
                                                                        else('30-39' if x <40
                                                                            else('40-49' if x < 50
                                                                                else('50-59' if x < 60
                                                                                    else '60-69' if x<70
                                                                                        else '70-79' if x<80
                                                                                            else '80+' if not pd.isnull(x) 
                                                                                                else '_Unknown'))))))

    florida['Age_group'] = florida['Age_group'].astype('category')
    florida['Age_group'].cat.categories
    order_cat = ['0-9',
                                                '10-19',
                                                '20-29',
                                                '30-39',
                                                '40-49',
                                                '50-59',
                                                '60-69',
                                                '70-79',
                                                '80+',
                                                '_Unknown']#[::-1]
    florida['Age_group'].cat.reorder_categories(order_cat, inplace=True)

    florida['Age_group'].cat.categories

    plt.rcParams['figure.figsize'] = [5.5, 4]
    SMALL_SIZE = 10.5
    MEDIUM_SIZE = 11
    MEDIUM_Plus= 11.5
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_Plus)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    return florida 

def florida_case_hosp_death(florida, gender):
    time = 'Case_'
    from matplotlib import dates

    plt.rcParams['figure.figsize'] = [13, 3.2]

    if gender != 'All':
        print(gender)
        florida = florida[florida['Gender']==gender]


    florida['all'] = 1
    vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
    lbs = ['COVID-19 Cases','COVID-19 Hospitalizations', 'COVID-19 Deaths']
    grpvar='Age_group'
    freq='7D'
    time=time
    newdf = florida

    fig, ax = plt.subplots(1,3)
    for i, v in enumerate(vrs):
        amt=v
        df = get_groupby_rolling_df(newdf, grpvar, freq, amt, time)
        ylb = 'Count' if i==0 else ''
        xlb = 'Positive Confirmed Date' if i==1 else ''
        df = df[df[grpvar]!='_Unknown']
        df[grpvar].cat.remove_unused_categories(inplace=True)

        p = df.groupby(grpvar)['rolling sum'].plot(ylabel=ylb, xlabel=xlb, ax=ax[i])
        
        #ax[i] = 
        graph_leg(p, '{}'.format(lbs[i]))
    # axes[0] = p[0]
    handles, labels = ax[2].get_legend_handles_labels()
    #fig.legend(reversed(handles), reversed(labels),labelspacing=0.04 ,bbox_to_anchor = (1.065,0.85), borderaxespad=0.)
    for i in range(3):
        date_fmt = '20%y-%m'
        formatter = dates.DateFormatter(date_fmt)
        ax[i].xaxis.set_major_locator(mdates.MonthLocator())
        ax[i].xaxis.set_major_formatter(formatter)
        plt.gcf().autofmt_xdate()
    plt.tight_layout()
    fig.legend(reversed(handles), reversed(labels),labelspacing=0.04 ,bbox_to_anchor = (1.065,0.85), borderaxespad=0.)
    return fig