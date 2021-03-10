# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# case fatality rate


# %%
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
pd.options.plotting.backend = "matplotlib"
from plotly.subplots import make_subplots


DATA_DIR = "../../../florida/data/"
time = 'Case_'
#print(time)
florida= pd.read_csv(os.path.join(
        DATA_DIR, 'florida_2021-01-05-15-35-01.csv'),index_col=False, parse_dates=[time, "ChartDate"])
florida["Age_group"] = florida["Age_group"].astype('category')
print(florida['Hospitalized'].value_counts())
print(florida['Hospitalized'].unique()) # 'YES'=1, else = 0
print(florida['Died'].unique()) # 'Yes'=1, nan = 0


# %%

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


# %%
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


# %%
florida[:3]


# %%
df = florida
df[(df[time]>=min_date) & (df[time]<=max_date)].count()
544915/804238

# %% [markdown]
# # all cases: died, hosp, covid
# 

# %%
import plotly.offline as pyo
pyo.init_notebook_mode()
pd.options.plotting.backend = "matplotlib"
import plotly.express as px


# %%
#from matplotlib import dates
#pd.options.plotting.backend = "plotly"
# Cases/Hospitalizations/Deaths

florida['all'] = 1
vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
lbs = ['COVID-19 Cases','COVID-19 Hospitalizations', 'COVID-19 Deaths']
vnames = ['Cases', 'Hospitalized', 'Died']
grpvar='Age_group'
freq='7D'
time=time
newdf = florida

figs = []
for i, v in enumerate(vrs):
    amt=v
    df = get_groupby_rolling_df(newdf, grpvar, freq, amt, time)
    ylb = 'Count' #if i==0 else ''
    xlb = 'Positive Confirmed Date' #if i==1 else ''
    df = df[df[grpvar]!='_Unknown']
    df[grpvar].cat.remove_unused_categories(inplace=True)

    df.groupby(grpvar)['rolling sum'].plot()#(ylabel=ylb, xlabel=xlb, ax=ax[i])
    df = df.reset_index()
    df = df.rename(columns={"Case_":"Date", 'rolling sum':vnames[i], 'Age_group':'Age Group'})
    fig = px.line(df.reset_index(), x='Date', y=vnames[i], color='Age Group')
    fig.update_layout(xaxis_title = xlb,
                    yaxis_title = ylb,
                    legend_title = "Age Group", 
                    yaxis={'categoryorder': 'total ascending'},
                    legend={'traceorder': 'reversed', 'font':{'size':12}},
                    font = dict(size=16)
                    )
    figs.append(fig)


    #fig.append_trace(p, 1, i)    

#     graph_leg(p, '{}'.format(lbs[i]))

# handles, labels = ax[2].get_legend_handles_labels()
# for i in range(3):
#     date_fmt = '20%y-%m'
#     formatter = dates.DateFormatter(date_fmt)
#     ax[i].xaxis.set_major_locator(mdates.MonthLocator())
#     ax[i].xaxis.set_major_formatter(formatter)
#     plt.gcf().autofmt_xdate()
# plt.tight_layout()
# fig.legend(reversed(handles), reversed(labels),labelspacing=0.04 ,bbox_to_anchor = (1.065,0.85), borderaxespad=0.)
# fig.savefig('img/{}'.format('cases.svg'), bbox_inches='tight')


# %%
figs[2]



# %% [markdown]
# # Bar plots for age ratios
# 
pd.options.plotting.backend = "matplotlib"


# %%
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
        newdf = florida.copy()
        min_date = "2020-04-01"
        max_date = "2020-02-01"
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
#fig.show()



# %%

florida['all'] = 1
vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
lbs = ['COVID-19 Cases','Hospitalizations', 'Deaths']
vnames = ['Cases', 'Hospitalizations', 'Deaths']
grpvar='Age_group'
freq='7D'

""" if gender != 'All':
    newdf = florida[florida['Gender']==gender]
else:
    newdf = florida

if time != 'Case_':
    if race!= 'All':
        newdf = newdf[newdf['race_ethnicity_combined']==race]
    if state != 'All':
        newdf = newdf[newdf['res_state']==state]
 """
newdf = florida.copy()
figs = []
for i, v in enumerate(vrs):
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
    
#fig.show()

# %%
figs[0]
#%%
m.columns
fig = px.bar(m, x='Date', y=vnames[i], color="Age Group", barmode = 'stack')


# %%
    
def get_bar_chart(amt, newdf, time, name, ax, it, grpvar='Age_group'):
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
plt.rcParams['figure.figsize'] = [13, 3.2]
fig, ax = plt.subplots(1,3)
vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
lbs = ['COVID-19 Cases','Hospitalizations', 'Deaths']

for i, v in enumerate(vrs):
    get_bar_chart(v, florida[florida['Age_group'] != '_Unknown'].copy(), time, '{}'.format(lbs[i]), ax[i], i)
    ax[i].set_xlabel( 'Positive Confirmed Date')

handles, labels = ax[2].get_legend_handles_labels()
plt.tight_layout()
fig.legend(reversed(handles), reversed(labels), labelspacing=0.04 ,bbox_to_anchor = (1.062,0.85), borderaxespad=0.)
fig.savefig('img/{}.svg'.format('age_ratios'), bbox_inches='tight')  

# %% [markdown]
# # Female ratio

# %%
plt.rcParams['figure.figsize'] = [13, 3.2]

vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
lbs = ['COVID-19 Cases', 'Hospitalizations','Deaths']
sv = ['all', 'hosp', 'death']

fig, ax = plt.subplots(1, 3)
plt.cla()

for i, v in enumerate(vrs):
    new_var = "f_{}".format(v)
    newdf[new_var] = 0
    newdf.loc[(newdf['Gender'] == 'Female') & (newdf[v] == 1), new_var] = 1
    df = get_groupby_rolling_df(newdf, grpvar, freq, new_var, time)
    df[new_var] = df['rolling sum']
    df_org = get_groupby_rolling_df(newdf, grpvar, freq, v, time)
    df[v] = df_org['rolling sum']
    df['female_prop'] = df[new_var] / df[v]
    cats = df[grpvar].cat.categories
    everylevel = pd.DataFrame(list(product(df.index.unique(), df[grpvar].cat.categories)), columns=[time, grpvar])
    df = df[df[v] >=5]
    df = df.merge(everylevel, on=[time, grpvar], how='outer')
    df.sort_values(by=time,inplace=True)
    df.index = df[time]
    df[grpvar] = df[grpvar].astype('category')
    df[grpvar].cat.reorder_categories(cats, inplace=True)
    df = remove_unkown(df, grpvar)
    p = df.groupby(grpvar)['female_prop'].plot(ax=ax[i], legend=False)

    graph_leg(p, '{}'.format(lbs[i]))
    ax[i].set_title("{}".format(lbs[i]))
    ax[i].set_xlabel("")
    ax[i].set_ylim(0,1)
    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(reversed(handles), reversed(labels), labelspacing=0.04 ,bbox_to_anchor = (1.065,0.85), borderaxespad=0.)
    plt.tight_layout()
ax[1].set_xlabel('Positive Confirmed Date')
for i in range(3):
    date_fmt = '20%y-%m'
    formatter = dates.DateFormatter(date_fmt)
    ax[i].xaxis.set_major_locator(mdates.MonthLocator())
    ax[i].xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
fig.savefig('img/{}'.format('gender_age_female.svg'.format(sv[i])), bbox_inches='tight')  


