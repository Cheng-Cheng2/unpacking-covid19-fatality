# %%
import argparse
import os
import pickle
import pprint
import warnings
import numpy as np
import pandas as pd
import glob
import json
import os
from utils import *
import plotly.express as px
#pd.options.plotting.backend = "plotly"
import pickle
IMGDIR = "cache_figs"

def get_datafiles_ready():
    min_date = '2020-03-26'
    max_date = "2021-02-01"
    CLEANDIR = "../../cleaned_data"
    florida = pd.read_csv(os.path.join(CLEANDIR, 'florida_2021-03-07-15-35-01.csv'),index_col=False, parse_dates=['Case_', "ChartDate"], dtype={'Age_group':'category'})
    
    print("Finished loading Florida data.")
    cdc = pd.read_csv(os.path.join(CLEANDIR, "cdc_02282021_total.csv"),index_col=False, parse_dates=['cdc_case_earliest_dt'],  dtype={'Age_group':'category', 'race_ethnicity_combined':'category'})  
    
    cdc['race_ethnicity_combined'].cat.rename_categories({'Asian, Non-Hispanic':'Asian',
                                 'Black, Non-Hispanic':'Black',
                                  'White, Non-Hispanic':'White',
                                   'Multiple/Other, Non-Hispanic ':'Multiple/Other',
                                    'Native Hawaiian/Other Pacific Islander, Non-Hispanic':'Native Hawaiian/Other Pacific Islander',
                                    'American Indian/Alaska Native, Non-Hispanic':'American Indian/Alaska Native'},
                                    inplace=True)
    print("Finished loading CDC data.")
    florida = florida[(florida['Case_']>=min_date) & (florida['Case_']<=max_date)]
    cdc = cdc[(cdc['cdc_case_earliest_dt']>= min_date) & (cdc['cdc_case_earliest_dt']<= max_date)]
    return florida, cdc

def florida_case_hosp_death(florida, gender='All', time='Case_', race='All', state='All'):
    fname = "time_{}_gender_{}_race_{}_state_{}_".format(time, gender, race, state)
    figs = []
    vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
    if not os.path.exists(os.path.join(IMGDIR,fname + vrs[0] +'.json')):
        florida['all'] = 1
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
            df.loc[df[vnames[i]] <=5, vnames[i]] = np.nan

            fig = px.line(df, x='Date', y=vnames[i], color='Age Group')

            showlegend = True #if i==2 else False
            fig.update_xaxes(
                dtick = 'M1',
                tickformat = "%b\n  %Y",
                tickangle = 30
            )
            fig.update_layout(xaxis_title = xlb,
                            yaxis_title = ylb,
                            legend_title = "Age Group", 
                            yaxis={'categoryorder': 'total ascending'},
                            legend={'traceorder': 'reversed', 'font':{'size':10}},
                            font = dict(size=16),
                            showlegend = showlegend,
                            margin={'l':0, 'r':0, 't':0, 'b':0},
                            width = 480,
                            height=300
                            )
            fn = os.path.join(IMGDIR,fname+vrs[i]+'.json')
            print("caching:", fn)
            with open(fn, 'w') as f:
            #     pickle.dump(html_bytes, f)
                f.write(fig.to_json())
            figs.append(fig)
    else:
        for v in vrs:     
            fn = os.path.join(IMGDIR, fname+v+'.json')
            print("loading:", fn)
            with open(fn, 'r') as f:
                fig = json.load(f)
                figs.append(fig)

    return figs


def florida_case_hosp_death_agg(florida, gender='All', time='Case_', race='All', state='All'):
    fname = "agg_time_{}_gender_{}_race_{}_state_{}_".format(time, gender, race, state)
    figs = []
    vrs = ['all', 'Hospitalized', 'Died'] # 'Died'
    if not os.path.exists(os.path.join(IMGDIR,fname + vrs[0] +'.json')):
        florida['all'] = 1
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
            df.loc[df[vnames[i]] <=5, vnames[i]] = np.nan

            fig = px.line(df, x='Date', y=vnames[i])

            showlegend = True #if i==2 else False
            fig.update_xaxes(
                dtick = 'M1',
                tickformat = "%b\n  %Y",
                tickangle = 30
            )
            fig.update_layout(xaxis_title = xlb,
                            yaxis_title = ylb,
                            legend_title = "Age Group", 
                            yaxis={'categoryorder': 'total ascending'},
                            legend={'traceorder': 'reversed', 'font':{'size':10}},
                            font = dict(size=16),
                            showlegend = showlegend,
                            margin={'l':0, 'r':0, 't':0, 'b':0},
                            width = 480,
                            height=300
                            )
            fn = os.path.join(IMGDIR,fname+vrs[i]+'.json')
            print("caching:", fn)
            with open(fn, 'w') as f:
            #     pickle.dump(html_bytes, f)
                f.write(fig.to_json())
            figs.append(fig)
    else:
        for v in vrs:     
            fn = os.path.join(IMGDIR, fname+v+'.json')
            print("loading:", fn)
            with open(fn, 'r') as f:
                fig = json.load(f)
                figs.append(fig)
    return figs

def age_distribution_plots(florida, gender='All', time='Case_', race='All', state='All'):
    fname = "age_time_{}_gender_{}_race_{}_state_{}_".format(time, gender, race, state)
    figs = []
    vrs = ['all', 'Hospitalized', 'Died'] # 'Died'

    if not os.path.exists(os.path.join(IMGDIR,fname + vrs[0] +'.json')):
        florida['all'] = 1

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
            
            fig.update_xaxes(
                    dtick = 'M1',
                    tickformat = "%b\n  %Y",
                    tickangle = 30
            )
            fig.update_layout(xaxis_title = xlb,
                            yaxis_title = ylb,
                            legend_title = "Age Group", 
                            legend_title_font_size = 12,
                            yaxis={'categoryorder': 'total ascending'},
                            legend={'traceorder': 'reversed', 'font':{'size':10}},
                            font = dict(size=16),
                            width = 480,
                            height=350
                            )    
            fn = os.path.join(IMGDIR,fname+vrs[i]+'.json')
            print("caching:", fn)
            with open(fn, 'w') as f:
            #     pickle.dump(html_bytes, f)
                f.write(fig.to_json())
            figs.append(fig)
    else:
        for v in vrs:     
            fn = os.path.join(IMGDIR, fname+v+'.json')
            print("loading:", fn)
            with open(fn, 'r') as f:
                fig = json.load(f)
                figs.append(fig)

    return figs

# 
# %%
if __name__ == '__main__':
    florida, cdc = get_datafiles_ready()

    race_cats = ['All']+cdc['race_ethnicity_combined'].dropna().unique().tolist()
    race_cats = [x for x in race_cats if (x != 'Missing')  and (x!='Unknown')]
    states_cats = cdc['res_state'].dropna().unique()#.tolist()
    states_cats = ['All']+[x for x in states_cats if (x != 'Missing') and (x!= 'Unknown')]

    for gender in ['All', 'Female', 'Male']:
        figs = florida_case_hosp_death_agg(florida, gender)
        figs = florida_case_hosp_death(florida, gender)
        figs = age_distribution_plots(florida, gender)


        for race in race_cats:
            for state in states_cats:
                figs = florida_case_hosp_death_agg(cdc, gender, 'cdc_case_earliest_dt', race, state)
                figs = florida_case_hosp_death(cdc, gender, 'cdc_case_earliest_dt', race, state)
                figs = age_distribution_plots(cdc, gender, 'cdc_case_earliest_dt', race, state)
