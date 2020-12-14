import os
import pickle
import pprint
from itertools import product

from scipy.interpolate import UnivariateSpline
from scipy import stats
from statsmodels.stats.weightstats import ztest


import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import dates

import numpy as np
from numpy.random import RandomState

import pandas as pd
from arch.bootstrap import MovingBlockBootstrap


"""Helper functions"""
def get_rolling_amount(grp, freq, amt, time):
    return grp.rolling(freq, on=time, min_periods=0)[amt].sum() #min_periods=0

def get_rolling_df(df, time='RESULT', freq=7, var='covid'):
    df = df[[time, var]]

    min_date = df[time].min()
    max_date = df[time].max()

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
    
    newdf = florida[florida['Hospitalized'] == 1].copy()  # among hospitalized
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


def load_florida_data(fpath='../data/florida/florida_2020-12-04-15-35-01.csv', timevar='Case_'):
    time = timevar
    florida= pd.read_csv(fpath, index_col=False, parse_dates=[time, "ChartDate"])
    florida["Age_group"] = florida["Age_group"].astype('category')
    
    florida['Case'].value_counts()
    florida['Died'].fillna(0, inplace=True)
    florida.loc[florida['Died']=='Yes', 'Died'] =1 


    florida.loc[florida['Hospitalized']!='YES', 'Hospitalized'] = 0
    florida.loc[florida['Hospitalized']=='YES', 'Hospitalized'] = 1
    florida.sort_values(by=time, inplace=True)
    florida[time] = florida[time].dt.date
    florida[time] = florida[time].astype('datetime64[ns]')

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
                 'Unknown']
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

    florida.rename(columns={'Age_group':'Old_age_group'}, inplace=True)

    florida['Age_group'] = florida['Age'].apply(lambda x: '0-9' if x < 10
                                                    else ('10-19' if x < 20 
                                                        else ('20-29' if x < 30 
                                                            else ('30-39' if x < 40 
                                                                else ('40-49' if x < 50 
                                                                    else ('50-59' if x < 60 
                                                                        else ('60-69' if x < 70 
                                                                            else('70-79'if x <= 80 
                                                                                else('80+' if not pd.isnull(x)
                                                                                     else '_Unknown')))))))))

    florida['Age_group'] = florida['Age_group'].astype('category')
    order_cat = ['0-9',
                 '10-19',
                 '20-29',
                 '30-39',
                 '40-49',
                 '50-59',
                 '60-69',
                 '70-79',
                 '80+',
                 '_Unknown']

    florida['Age_group'].cat.reorder_categories(order_cat, inplace=True)
    return florida


def print_support(florida):
    grpvar = 'Age_group'
    freq = '7D'
    amt = 'Died'
    timevar = 'Case_'

    hfr_df, age_hfr = get_HFR(florida, grpvar, freq, timevar, numfreq=7)

    print('aggregate')
    print(hfr_df[hfr_df.index.isin(['2020-04-01', '2020-04-15', '2020-07-15', '2020-11-01'])]['Hospitalized'])
    print(age_hfr[age_hfr.index.isin(['2020-04-01', '2020-04-15', '2020-07-15', '2020-11-01'])][['Hospitalized', 'Died', 'Age_group']])


def load_national_data(timevar='cdc_report_dt', fpath='../data/national/covid_national_9_30.csv', verbose=False):
    df = pd.read_csv(fpath, 
                     parse_dates=['cdc_report_dt', 'pos_spec_dt', 'onset_dt'])

    time = timevar
    df = df[df['current_status'] == 'Laboratory-confirmed case'].copy()

    percent_missing = df.isnull().sum() * 100 / len(df)
    if verbose:
        print('percent missing:', percent_missing)

    df['age_group'] = df['age_group'].astype('category')
    df['age_group'].cat.categories
    df.loc[df['age_group'].isnull(), 'age_group'] = 'Unknown'
    df['age_group'].cat.rename_categories({'Unknown': '_' + 'Unknown'}, inplace=True)
    df['age_group'].cat.categories
    df['age_group'].cat.rename_categories({'0 - 9 Years':'0-9', 
                                           '10 - 19 Years':'10-19',
                                           '20 - 29 Years':'20-29',
                                           '30 - 39 Years':'30-39',
                                           '40 - 49 Years':'40-49', 
                                           '50 - 59 Years':'50-59',
                                           '60 - 69 Years':'60-69',
                                           '70 - 79 Years':'70-79',
                                           '80+ Years':'80+'}, inplace=True)

    df.loc[df['hosp_yn']!='Yes', 'hosp_yn'] = 0
    df.loc[df['hosp_yn']=='Yes', 'hosp_yn'] = 1

    df.loc[df['death_yn']!='Yes', 'death_yn'] = 0
    df.loc[df['death_yn']=='Yes', 'death_yn'] = 1

    # rename everything to match florida format
    cleaned_country = df.rename(columns={'age_group':'Age_group', 
                                        'sex': 'Gender',
                                        'hosp_yn': 'Hospitalized',
                                        'death_yn': 'Died',
                                        'current_status': 'Case',
                                        time: 'Case_'})
    cleaned_country['Case'] = 1
    cleaned_country.sort_values(by='Case_', inplace=True)
    florida = cleaned_country.copy()

    order_cat = ['0-9',
                 '10-19',
                 '20-29',
                 '30-39',
                 '40-49',
                 '50-59',
                 '60-69',
                 '70-79',
                 '80+',
                 '_Unknown']
    florida['Age_group'].cat.reorder_categories(order_cat, inplace=True)

    return florida


def get_combined_df(florida, min_date, max_date, 
                    grpvar='Age_group', freq='7D', amt='Died', timevar='Case_', 
                    verbose=False):
    # get HFRs in aggregate and per age group
    hfr_df, age_hfr = get_HFR(florida, grpvar, freq, timevar, numfreq=7)

    if verbose:
        print('=========== counts on days of interest ==============')
        print(hfr_df[hfr_df.index.isin(days_of_interest)]['Hospitalized'])
        print(age_hfr[age_hfr.index.isin(days_of_interest)][['Hospitalized', 'Died', 'Age_group']])
        print('=====================================================')

    hfr_list = hfr_df['HFR'].reset_index().to_dict('list')
    new_idx = hfr_list['Case_']
    values = {'aggregate': hfr_list['HFR']}

    for age_group in age_hfr['Age_group'].unique():
        age_df = age_hfr[age_hfr['Age_group'] == age_group]
        age_list = age_df['HFR'].reset_index().to_dict('list')
        
        if new_idx != age_list['Case_']:
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
    

def fit_splines(combined_df, TAG='national'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    residuals = {}
    trends = {}
    xs = {}
    for col in combined_df.columns:
        if col == 'day_ct':
            continue
        elif col == 'aggregate':
            i = 0
        else:
            i = 1
        this_df = combined_df[combined_df[col].notna()]
        spline = UnivariateSpline(this_df['day_ct'], this_df[col])#, s=0.1)

        trends[col] = spline(this_df['day_ct'])
        residuals[col] = spline(this_df['day_ct']) - this_df[col]
        xs[col] = (this_df['day_ct'], this_df.index)
        
        p = ax[i].plot(this_df.index, this_df[col], '-', label=col)
        ax[i].plot(this_df.index, spline(this_df['day_ct']), '--', color=p[0].get_color())
        plt.legend()
        
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])

    ax[0].legend()
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.suptitle('Age-stratified trend approximated by smoothing splines')

    date_fmt = '20%y-%m'
    formatter = dates.DateFormatter(date_fmt)
    ax[0].xaxis.set_major_locator(dates.MonthLocator())
    ax[0].xaxis.set_major_formatter(formatter)
    ax[1].xaxis.set_major_locator(dates.MonthLocator())
    ax[1].xaxis.set_major_formatter(formatter)
    
    if 'national' in TAG:
        dname = 'CDC Report Date'
        folder = 'img_country'
    else:
        assert('florida' in TAG)
        dname = 'Positive Confirmed Date'
        folder = 'img'
    ax[0].set_xlabel(dname)
    ax[1].set_xlabel(dname)
    ax[0].set_ylabel('HFR')
    ax[1].set_ylabel('HFR')

    plt.gcf().autofmt_xdate()

    plt.savefig(f'{folder}/{TAG}_trends.svg')

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


def t_test_for_gender(male_boot, female_boot, decreases_of_interest, z_test=False):
    assert(male_boot.keys() == female_boot.keys())

    t_pvals = {}
    for col in male_boot:
        t_pvals[col] = {}
        if col == 'day_ct' or col == '_Unknown':
            continue

        for dec, male, female in zip(decreases_of_interest, male_boot[col], female_boot[col]):
            if z_test:
                stat, pval = ztest(male, female)
            else:
                stat, pval = stats.ttest_ind(male, female, equal_var=False)
            t_pvals[col][dec] = pval
    return t_pvals


def summarize_boot_ests(boot_trends, days_of_interest, decreases_of_interest, xs, min_date, TAG='national'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    day_ests = {}
    dec_ests = {}
    dec_replicates = {}
    for col in boot_trends:
        if col in ['day_ct', '_Unknown', '0-9', '10-19']:
            continue
        elif col == 'aggregate':
            i = 0
        else:
            i = 1
            
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
            
        p = ax[i].plot(xs[col][1], median, label=col)
        ax[i].fill_between(xs[col][1], lower, upper, alpha=0.2, color=p[0].get_color())

    ax[0].set_ylim([0,1])
    ax[1].set_ylim([0,1])
    fig.suptitle('Uncertaity estimates around trends')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    date_fmt = '20%y-%m'
    formatter = dates.DateFormatter(date_fmt)
    ax[0].xaxis.set_major_locator(dates.MonthLocator())
    ax[0].xaxis.set_major_formatter(formatter)
    ax[1].xaxis.set_major_locator(dates.MonthLocator())
    ax[1].xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()

    if 'national' in TAG:
        dname = 'CDC Report Date'
        folder = 'img_country'
    else:
        assert('florida' in TAG)
        dname = 'Positive Confirmed Date'
        folder = 'img'
    ax[0].set_xlabel(dname)
    ax[1].set_xlabel(dname)
    ax[0].set_ylabel('HFR')
    ax[1].set_ylabel('HFR')

    plt.savefig(f'{folder}/{TAG}_unc_est.svg')
    
    return day_ests, dec_ests, dec_replicates


def summarize_days_of_interest(day_ests, dec_ests, days_of_interest, TAG='national'):
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


if __name__ == '__main__':
    # data from: https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data/vbim-akqf
    data_fpath = '../data/national/COVID_Cases_Restricted_Detailed_12042020.csv'
    
    # data from: https://www.arcgis.com/home/item.html?id=4cc62b3a510949c7a8167f6baa3e069d
    florida_fpath = '../data/florida/florida_2020-12-04-15-35-01.csv'

    days_of_interest = [
        '2020-04-01',
        '2020-04-15',
        '2020-07-15',
        '2020-11-01',
    ]

    decreases_of_interest = [
        ('2020-04-01', '2020-11-01'),
        ('2020-04-15', '2020-07-15'),
        ('2020-07-15', '2020-11-01'),
    ]

    MIN_DATE = '2020-04-01'
    MAX_DATE = '2020-11-01'
    FIG_DIR = '../results/'
    verbose = False
    per_state = False
    gender = False
    z_test = False
    only_18_plus = False

    def analyze_data(florida, TAG, return_replicates=False):
        ## GET HFRS
        combined_df = get_combined_df(florida, MIN_DATE, MAX_DATE, 
                        grpvar='Age_group', freq='7D', amt='Died', timevar='Case_', 
                        verbose=verbose)

        ## CUBIC SPLINES
        # fit cubic splines and get residuals
        combined_df = combined_df[combined_df.index <= MAX_DATE]
        combined_df = combined_df[combined_df.index >= MIN_DATE]

        resid_df, trends, xs = fit_splines(combined_df, TAG=TAG)

        ## Block boostrap
        boot_trends = get_bootstrap_estimates(resid_df, trends, xs, seed=1234, verbose=False)
        day_ests, dec_ests, dec_replicates = summarize_boot_ests(boot_trends, days_of_interest, decreases_of_interest, xs, MIN_DATE, TAG=TAG)
        all_est_df = summarize_days_of_interest(day_ests, dec_ests, days_of_interest, TAG=TAG)
        
        pd.set_option("display.max_columns", 101)

        if return_replicates:
            return all_est_df, dec_replicates
        return all_est_df

    
    if per_state:
        pos_spec_states = ["MA", "AR", "KS", "WA", "LA", "NY", "PA", "NV", "UT", "MN", "NC"]
        cdc_report_states = ["CA", "AZ", "GA", "TN"]

        print('loading national_data w/ cdc_report_dt...')
        cdc_report_df = load_national_data(timevar='cdc_report_dt', fpath=data_fpath, verbose=verbose)

        print(f'====================== NATIONAL ========================')
        florida = cdc_report_df
        analyze_data(florida, 'national')

        for cur_state in cdc_report_states:
            print(f'====================== {cur_state} ========================')
            florida = cdc_report_df[cdc_report_df['res_state'] == cur_state]
            analyze_data(florida, cur_state)
        
        print('loading national_data w/ pos_spec_dt...')
        pos_spec_df = load_national_data(timevar='pos_spec_dt', fpath=data_fpath, verbose=verbose)
        for cur_state in pos_spec_states:
            print(f'====================== {cur_state} ========================')
            florida = pos_spec_df[pos_spec_df['res_state'] == cur_state]
            analyze_data(florida, cur_state)
    elif gender:
        print('=============================== CHECK GENDER DIFFERENCES =====================================')

        florida = load_florida_data(fpath='../data/florida/florida_2020-12-04-15-35-01.csv', timevar='Case_')

        print('-------------------- FLORIDA MALE --------------------')
        florida_male = florida[florida['Gender'] == 'Male']
        florida_est_male, fl_male_boot = analyze_data(florida_male, 'florida_fdoh', return_replicates=True)

        print_support(florida_male)
        print(florida_est_male)

        print('-------------------- FLORIDA FEMALE ------------------------')
        florida_female = florida[florida['Gender'] == 'Female']
        florida_est_female, fl_female_boot = analyze_data(florida_female, 'florida_fdoh', return_replicates=True)
        
        print_support(florida_female)
        print(florida_est_female)

        print('-------------------- T-Test ------------------------')
        t_scores_fl = t_test_for_gender(fl_male_boot, fl_female_boot, decreases_of_interest, z_test=True)
        pprint.pprint(t_scores_fl)

        # national but with NJ, IL, and CT removed
        print('==================== CLEANED NATIONAL (no NJ, IL, CT) ====================')
        cdc_report_df = load_national_data(timevar='cdc_report_dt', fpath=data_fpath, verbose=verbose)
        national = cdc_report_df[~cdc_report_df['res_state'].isin(['CT','IL','NJ'])]

        print('-------------------- NATIONAL MALE --------------------')
        national_male = national[national['Gender'] == 'Male']
        national_est_male, nat_male_boot = analyze_data(national_male, 'national_cleaned', return_replicates=True)
        print_support(national_male)
        print(national_est_male)

        print('-------------------- NATIONAL FEMALE --------------------')
        national_female = national[national['Gender'] == 'Female']
        national_est_female, nat_female_boot = analyze_data(national_female, 'national_cleaned', return_replicates=True)
        print_support(national_female)
        print(national_est_female)

        print('-------------------- T-Test ------------------------')
        t_scores_nat = t_test_for_gender(nat_male_boot, nat_female_boot, decreases_of_interest, z_test=True)
        pprint.pprint(t_scores_nat)

        print('======================== COMBINED FOR PAPER =======================')
        for_paper = ['2020-04-01', '2020-11-01', '04-01 to 11-01']
        multindex = list(zip(['florida'] * 3 + ['national'] * 3, for_paper * 2))
        multindex = pd.MultiIndex.from_tuples(multindex, names=['loc', 'dates'])
        
        paper_df_male = pd.concat([florida_est_male[for_paper], national_est_male[for_paper]], axis=1)
        # paper_df = pd.concat([florida_est[for_paper], national_est[for_paper]], axis=0)
        paper_df_male.columns = multindex
        print(f'------------ male {for_paper} ----------')
        print(paper_df_male.to_latex(index=True))

        paper_df_female = pd.concat([florida_est_female[for_paper], national_est_female[for_paper]], axis=1)
        # paper_df = pd.concat([florida_est[for_paper], national_est[for_paper]], axis=0)
        paper_df_female.columns = multindex
        print(f'------------ female {for_paper} ----------')
        print(paper_df_female.to_latex(index=True))

        for_paper = ['2020-04-15', '2020-07-15', '04-15 to 07-15']
        multindex = list(zip(['florida'] * 3 + ['national'] * 3, for_paper * 2))
        multindex = pd.MultiIndex.from_tuples(multindex, names=['loc', 'dates'])
        
        paper_df_male = pd.concat([florida_est_male[for_paper], national_est_male[for_paper]], axis=1)
        # paper_df = pd.concat([florida_est[for_paper], national_est[for_paper]], axis=0)
        paper_df_male.columns = multindex
        print(f'------------ male {for_paper} ----------')
        print(paper_df_male.to_latex(index=True))

        paper_df_female = pd.concat([florida_est_female[for_paper], national_est_female[for_paper]], axis=1)
        # paper_df = pd.concat([florida_est[for_paper], national_est[for_paper]], axis=0)
        paper_df_female.columns = multindex
        print(f'------------ female {for_paper} ----------')
        print(paper_df_female.to_latex(index=True))

    else:
        print('==================== FLORIDA ====================')
        florida = load_florida_data(fpath='../data/florida/florida_2020-12-04-15-35-01.csv', timevar='Case_')
        print_support(florida)

        if only_18_plus:
            florida = florida[florida['Age'] >= 18]

        florida_est = analyze_data(florida, 'florida_fdoh')
        print(florida_est)

        # national but with NJ, IL, and CT removed
        print('==================== CLEANED NATIONAL (no NJ, IL, CT) ====================')
        cdc_report_df = load_national_data(timevar='cdc_report_dt', fpath=data_fpath, verbose=verbose)
        national = cdc_report_df[~cdc_report_df['res_state'].isin(['CT','IL','NJ'])]
        print_support(national)

        if only_18_plus:
            national = national[~national['Age_group'].isin(['0-9', '10-19'])]

        national_est = analyze_data(national, 'national_cleaned')
        print(national_est)

        
        print('====================================================================')

        # print('-------------------------------------------------')
        # print(florida_est.to_latex(index=False))
        # print('-------------------------------------------------')
        
        # print('-------------------------------------------------')
        # print(national_est.to_latex(index=False))
        # print('-------------------------------------------------')
        for_paper = ['2020-04-01', '2020-11-01', '04-01 to 11-01']
        # for_paper = ['2020-04-15', '2020-07-15', '04-15 to 07-15']
        # for_paper = ['2020-07-15', '2020-11-01', '07-15 to 11-01']
        multindex = list(zip(['florida'] * 3 + ['national'] * 3, for_paper * 2))
        multindex = pd.MultiIndex.from_tuples(multindex, names=['loc', 'dates'])
        
        paper_df = pd.concat([florida_est[for_paper], national_est[for_paper]], axis=1)
        # paper_df = pd.concat([florida_est[for_paper], national_est[for_paper]], axis=0)
        paper_df.columns = multindex
        print(paper_df.to_latex(index=True))

