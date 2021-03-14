# %%
from datetime import time
import os
import pickle
import pprint
from itertools import product

from numpy.core.numeric import NaN

from scipy.interpolate import UnivariateSpline

import numpy as np
from numpy.random import RandomState

import pandas as pd
from arch.bootstrap import MovingBlockBootstrap
from plotting_utils import *
import pdb

"""Helper functions"""
def get_rolling_amount(grp, freq, amt, time):
    return grp.rolling(freq, on=time, min_periods=0)[amt].sum() #min_periods=0

def get_rolling_df(df, time='RESULT', freq=7, var='covid'):
    df = df[[time, var]]

    min_date = df[time].min()
    max_date = df[time].max()
    #pdb.set_trace()
    dts = pd.date_range(min_date, max_date)
    var_val = [0]*len(dts)
    everylevel = pd.DataFrame({time:dts, var:var_val})
    df = df.merge(everylevel, on=[time, var], how='outer')
    df.sort_values(by=time,inplace=True)

    freqd = '{}D'.format(freq)
    df.set_index(time, inplace=True)
    df = df.rolling(freqd, min_periods=0)[var].sum().reset_index()
    df = df.set_index(time)
    df = df.groupby(time)[var].max().reset_index()
    df = df.set_index(time)

    df[var].fillna(0, inplace=True)
    df[var] = df[var]/freq
    df.sort_values(by=time,inplace=True)
    return df

def get_groupby_rolling_df(df, grpvar='Age group', freq='7D', amt="Died", time='Calendar date'):
    cats = df[grpvar].cat.categories
    out='rolling sum'
    df = df[[time, grpvar, amt]]
    # merge on precomputed
    min_date = df[time].min()
    max_date = df[time].max()
    dts = pd.date_range(min_date, max_date)
    everylevel = pd.DataFrame(list(product(dts.unique(), df[grpvar].cat.categories)), columns=[time, grpvar])
    everylevel[amt] = 0
    df = df.merge(everylevel, on=[time, grpvar, amt], how='outer')
    df.sort_values(by=time,inplace=True)
    df[grpvar] = df[grpvar].astype('category')
    df[grpvar].cat.reorder_categories(cats, inplace=True)
    df[out]= df.groupby([grpvar],  as_index=False, group_keys=False) \
                               .apply(get_rolling_amount, freq, amt, time)
    df = df.groupby([grpvar, time])[out].max().reset_index()
    df[out].fillna(0, inplace=True)
    df = df[[grpvar, time, out]]
    df[out] = df[out]/7
    df = df.set_index(time)
    return df

def get_HFR(florida, grpvar, freq, timevar, numfreq=7):
    time = timevar
    max_date = florida[time].max()
    min_date = florida[time].min()
    print("time", time, 'min_date:', min_date, "max_date:", max_date)
    #pdb.set_trace()
    if pd.isnull(min_date) or pd.isnull(max_date):
        pdb.set_trace()
    newdf = florida[florida['Hospitalized'] == 1].copy()  # among hospitalized
    if newdf is None or len(newdf)==None:
        return None, None
    died_roll = get_rolling_df(newdf, time, freq=numfreq, var='Died').copy()
    hosp_roll = get_rolling_df(newdf, time, freq=numfreq, var='Hospitalized').copy()

    hfr_df = died_roll.copy()
    hfr_df['Hospitalized'] = hosp_roll['Hospitalized']
    hfr_df['HFR'] = hfr_df['Died'] / hfr_df['Hospitalized']
    
    newdf = florida[florida['Hospitalized'] ==1].copy()
    age_died = get_groupby_rolling_df(newdf, grpvar, freq, 'Died', time).copy()
    age_died['Died'] = age_died['rolling sum']

    age_hosp = get_groupby_rolling_df(newdf, grpvar, freq, 'Hospitalized', time).copy()
    age_hosp['Hospitalized'] = age_hosp['rolling sum']

    age_hfr = age_died.copy()
    age_hfr['Hospitalized'] = age_hosp['Hospitalized']
    age_hfr['HFR'] = age_hfr['Died'] / age_hfr['Hospitalized']

    return hfr_df, age_hfr


def triple_to_str(row, col):  # make readable with 2 sig figs
        r = row[col]
        s = "%.2g (%.2g, %.2g)" % (r[0], r[1], r[2])
        return s


def get_support(florida, dates_of_interest, min_deaths=2, timevar='Case_'):
    grpvar = 'Age_group'
    freq = '7D'
    #timevar = 'Case_'

    hfr_df, age_hfr = get_HFR(florida, grpvar, freq, timevar, numfreq=7)
    if hfr_df is None or age_hfr is None:
        return None
    age_hfr = age_hfr[age_hfr.index.isin(dates_of_interest)][['Hospitalized', 'Died', 'Age_group']]
    min_support = age_hfr.groupby('Age_group')['Died'].min()
    min_support = (min_support > min_deaths).reset_index()
    min_support_ages = [age for age, died in zip(min_support['Age_group'].tolist(), min_support['Died'].tolist()) if died]
    min_support_ages = min_support_ages + ['aggregate']
    return min_support_ages


def get_combined_df(florida, min_date, max_date, 
                    grpvar='Age_group', freq='7D', amt='Died', timevar='Case_', 
                    verbose=False):
    # get HFRs in aggregate and per age group
    hfr_df, age_hfr = get_HFR(florida, grpvar, freq, timevar, numfreq=7)
    if hfr_df is None or age_hfr is None:
        return None

    if verbose:
        print('=========== counts on days of interest ==============')
        print(hfr_df[hfr_df.index.isin(days_of_interest)]['Hospitalized'])
        print(age_hfr[age_hfr.index.isin(days_of_interest)][['Hospitalized', 'Died', 'Age_group']])
        print('=====================================================')

    hfr_list = hfr_df['HFR'].reset_index().to_dict('list')
    new_idx = hfr_list[timevar]
    values = {'aggregate': hfr_list['HFR']}

    for age_group in age_hfr['Age_group'].unique():
        age_df = age_hfr[age_hfr['Age_group'] == age_group]
        age_list = age_df['HFR'].reset_index().to_dict('list')
        
        if new_idx != age_list[timevar]:
            import pdb; pdb.set_trace()
        else:
            values[age_group] = age_list['HFR']

    combined_df = pd.DataFrame(values, index=new_idx)
    combined_df = combined_df[(combined_df.index >= pd.to_datetime(min_date))]
    combined_df = combined_df[(combined_df.index <= pd.to_datetime(max_date))]

    days_since_min = (combined_df.index - pd.Timestamp(min_date)).days
    combined_df['day_ct'] = days_since_min
    combined_df = combined_df.drop('_Unknown', axis=1)
    return combined_df
    

def fit_splines(combined_df):
    residuals = {}
    trends = {}
    xs = {}
    for col in combined_df.columns:
        if col in ['day_ct', '_Unknown', '0-9', '10-19']:
            continue
        this_df = combined_df[combined_df[col].notna()]
        spline = UnivariateSpline(this_df['day_ct'], this_df[col])#, s=0.1)

        trends[col] = spline(this_df['day_ct'])
        residuals[col] = spline(this_df['day_ct']) - this_df[col]
        xs[col] = (this_df['day_ct'], this_df.index)

    residuals['day_ct'] = combined_df['day_ct']
    residuals['Case_'] = combined_df.index
    resid_df = pd.DataFrame(residuals)
    resid_df = resid_df.set_index('Case_')

    return resid_df, trends, xs


def get_bootstrap_estimates(resid_df, trends, xs, seed=1234, verbose=False):
    rs = RandomState(seed)
    boot_trends = {}
    for col in resid_df.columns:
        if col == 'day_ct':
            continue

        this_df = resid_df[['day_ct', col]]
        this_df = resid_df[resid_df[col].notna()]
        if verbose:
            print(col)
        bs = MovingBlockBootstrap(7, this_df, random_state=rs)

        for resid in bs.bootstrap(1000):
            bs_df = resid[0][0]
            new_data = bs_df[col] + trends[col]
            spline = UnivariateSpline(xs[col][0], new_data.tolist())
            boot_trends[col] = boot_trends.get(col, []) + [spline(xs[col][0])]
    return boot_trends


def summarize_boot_ests(boot_trends, days_of_interest, decreases_of_interest, min_date):
    day_ests = {}
    dec_ests = {}
    dec_replicates = {}
    for col in boot_trends:
        if col in ['day_ct', '_Unknown', '0-9', '10-19']:
            continue
            
        lower = np.quantile(boot_trends[col], 0.025, axis=0)
        upper = np.quantile(boot_trends[col], 0.975, axis=0)
        median = np.quantile(boot_trends[col], 0.5, axis=0)
        
        dt_to_daynum = {}
        for dy in days_of_interest:
            day_num = (pd.Timestamp(dy) - pd.Timestamp(min_date)).days
            day_ests[col] = day_ests.get(col, []) + [(median[day_num], lower[day_num], upper[day_num])]
            dt_to_daynum[dy] = day_num

        for pair in decreases_of_interest:    
            d1 = dt_to_daynum[pair[0]]
            d2 = dt_to_daynum[pair[1]]
            arr = np.array(boot_trends[col])
            p1 = arr[:, d1]
            p2 = arr[:, d2]
            dec = (p2 - p1) / p1
            dec_ests[col] = dec_ests.get(col, []) + [(np.quantile(dec, 0.5), 
                                                      np.quantile(dec, 0.025), 
                                                      np.quantile(dec, 0.975))]
            dec_replicates[col] = dec_replicates.get(col, []) + [dec]

    return day_ests, dec_ests, dec_replicates


def summarize_days_of_interest(day_ests, dec_ests, days_of_interest, decreases_of_interest):
    # compute decreases between dates of interest
    dec_ests['Time range'] = list([' to '.join([de.replace('2020-', '') for de in dec]) for dec in decreases_of_interest])
    dec_df = pd.DataFrame(dec_ests)
    age_groups = dec_df.columns.tolist()
    dec_df = pd.DataFrame(dec_df.values.T)
    dec_df.columns = dec_df.iloc[-1]
    dec_df['Age_group'] = age_groups
    dec_df = dec_df.set_index('Age_group')

    dec_df = dec_df.drop('Time range')
    dec_df = pd.DataFrame(dec_df.to_dict())
    
    # compute estimates on days of interest
    day_ests['Day'] = days_of_interest

    est_df = pd.DataFrame(day_ests)
    age_groups = est_df.columns.tolist()
    est_df = pd.DataFrame(est_df.values.T)
    est_df.columns = est_df.iloc[-1]
    est_df['Age_group'] = age_groups
    est_df = est_df.set_index('Age_group')

    est_df = est_df.drop('Day')
    est_df = pd.DataFrame(est_df.to_dict())

    all_est_df = pd.concat([est_df, dec_df], axis=1)

    # for col in days_of_interest:
    for col in all_est_df:
        all_est_df[col] = all_est_df.apply(lambda x: triple_to_str(x, col), axis=1)
    return all_est_df


def compute_hfr_estimates(national, florida, days_of_interest, filters=None, update_florida=True):
    assert(len(days_of_interest) == 2)
    decreases_of_interest = [(days_of_interest[0], days_of_interest[1])]

    MIN_DATE = '2020-04-01'
    MAX_DATE = '2021-02-01'
    verbose = False

    def analyze_data(florida, TAG, return_replicates=False, timevar='Case_'):
        ## GET HFRS
        combined_df = get_combined_df(florida, MIN_DATE, MAX_DATE, 
                        grpvar='Age_group', freq='7D', amt='Died', timevar=timevar, 
                        verbose=verbose)
        if combined_df is None:
            return None                        

        ## CUBIC SPLINES

        # fit cubic splines and get residuals
        combined_df = combined_df[combined_df.index <= MAX_DATE]
        combined_df = combined_df[combined_df.index >= MIN_DATE]

        resid_df, trends, xs = fit_splines(combined_df)

        ## Block boostrap
        boot_trends = get_bootstrap_estimates(resid_df, trends, xs, seed=1234, verbose=False)
        day_ests, dec_ests, dec_replicates = summarize_boot_ests(boot_trends, days_of_interest, decreases_of_interest, MIN_DATE)
        all_est_df = summarize_days_of_interest(day_ests, dec_ests, days_of_interest, decreases_of_interest)
        
        pd.set_option("display.max_columns", 101)

        if return_replicates:
            return all_est_df, dec_replicates
        return all_est_df

    if filters is not None:
        for k, v in filters.items():
            if update_florida:
                
                florida = florida[florida[k] == v]
                
            else:
                #pdb.set_trace() 
                national = national[national[k] == v]

        if update_florida and (florida is None or len(florida)==0):
            return None 
        if (update_florida==False) and (national is None or len(national)==0):
            return None


    if update_florida == True:
        print('==================== FLORIDA ====================')
        min_support_ages = get_support(florida, days_of_interest)
        if min_support_ages is None:
            return None
        florida_est = analyze_data(florida, 'florida_fdoh', timevar='Case_')
        florida_est = florida_est[florida_est.index.isin(min_support_ages)]
        print(florida_est)
        return florida_est
    else:
        print('==================== NATIONAL ====================')
        min_support_ages = get_support(national, days_of_interest, 2, 'cdc_case_earliest_dt')
        if min_support_ages is None:
            return None
        national_est = analyze_data(national, 'national', timevar='cdc_case_earliest_dt')
        national_est = national_est[national_est.index.isin(min_support_ages)]
        print(national_est)
        return national_est
    #return florida_est, national_est


# %%
# from block_bootstrap_hfr import load_florida_data, load_national_data
# data_fpath = '../data/national/covid_national_12_31.csv'
# florida_fpath = '../data/florida/florida_2021-01-03-15-35-01.csv'

# #florida = load_florida_data(fpath=florida_fpath, timevar='Case_')
# #national = load_national_data(timevar='cdc_report_dt', fpath=data_fpath, verbose=False)

# florida, national = get_datafiles_ready()

# days_of_interest = ['2020-04-01', '2020-12-01']
# filters = {'res_state': 'MI', 'Gender': 'Female'}
# florida_est, national_est = compute_hfr_estimates(national, florida, days_of_interest, filters=None)

# %%
