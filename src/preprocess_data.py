#%% 
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
from utils import *

CDC_DATE = '08312021'
FLORIDA_DATE = '2021-08-31-15-35-01'
CLEANDIR = '../data/cleaned'
FLORIDA_DATA_DIR = '../data'
CDC_DATA_DIR = '../data'

def get_florida_data_ready():
    DATA_DIR = FLORIDA_DATA_DIR
    time = 'Case_'
    #print(time)
    florida= pd.read_csv(os.path.join(
            DATA_DIR, f'florida_{FLORIDA_DATE}.csv'),index_col=False, parse_dates=[time, "ChartDate"])
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

    return florida 


def get_cdc_data_ready():
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


    data_dir = CDC_DATA_DIR
    country = pd.read_csv(os.path.join(data_dir, f'cdc_{CDC_DATE}_total.csv'), parse_dates=['cdc_case_earliest_dt'])


    print('all cases', country.shape)
    percent_missing = country.isnull().sum() * 100 / len(country)
    percent_missing
    #df = country[country['current_status']=='Laboratory-confirmed case']
    #df = country.copy()
    #print(df.shape)

    min_date = pd.Timestamp('2020-04-01')
    max_date = pd.Timestamp('2020-12-01')

    test()
    time = "cdc_case_earliest_dt"

        # should we include probable case??
    #time = 'cdc_case_earliest_dt'
    df = country[country['current_status']=='Laboratory-confirmed case']
    #percent_missing = df.isnull().sum() * 100 / len(df)
    #print('before:', percent_missing)

    if False: # takes time to inpute 'pos_spec_dt'
        time = 'pos_spec_dt'
        df[time] = df.apply(lambda r: r[time] if not pd.isnull(r[time])
                                else (r['cdc_case_earliest_dt']), axis=1)
        #pd.to_csv(os.path.joint(data_dir, 'country_positive_test'.csv), index=False)
        df.to_csv(os.path.join(data_dir, 'country_positive_test.csv'), index=False)


    #df = pd.read_csv(os.path.join(data_dir, 'country_positive_test.csv'), parse_dates=['pos_spec_dt', 'cdc_case_earliest_dt'])

    # rename age_groups
    #print("pos inputed missingness")
    percent_missing = df.isnull().sum() * 100 / len(df)
    print(percent_missing)
    df['age_group'] = df['age_group'].astype('category')
    df['age_group'].cat.categories
    df.loc[df['age_group'].isnull(), 'age_group'] = 'Missing'
    df['age_group'].cat.rename_categories({'Missing': '_' + 'Unknown'}, inplace=True)
    df['age_group'].cat.categories

    #df['age_group']
    df['age_group'].cat.rename_categories({'0 - 9 Years':'0-9', 
                                        '10 - 19 Years':'10-19',
                                        '20 - 29 Years':'20-29',
                                        '30 - 39 Years':'30-39',
                                        '40 - 49 Years':'40-49', 
                                        '50 - 59 Years':'50-59',
                                        '60 - 69 Years':'60-69',
                                        '70 - 79 Years':'70-79',
                                        '80+ Years':'80+'}, inplace=True)
    df.loc[df['res_state']=='NYC', 'res_state'] = 'NY'
    #

        # hosp
    df.loc[df['hosp_yn']!='Yes', 'hosp_yn'] = 0
    df.loc[df['hosp_yn']=='Yes', 'hosp_yn'] = 1


    # death
    df.loc[df['death_yn']!='Yes', 'death_yn'] = 0
    df.loc[df['death_yn']=='Yes', 'death_yn'] = 1



    print(df['age_group'].value_counts())
    print(df['hosp_yn'].value_counts())
    print(df['death_yn'].value_counts())


    cleaned_country = df.rename(columns={'age_group':'Age_group', 
                                        'sex': 'Gender',
                                        'hosp_yn': 'Hospitalized',
                                        'death_yn': 'Died',
                                        'current_status': 'Case'})
    cleaned_country['Case'] = 1

    #### se;lect  the date to use

    time = 'cdc_case_earliest_dt'
    cleaned_country.sort_values(by=time, inplace=True)
    print(cleaned_country.columns)
    florida = cleaned_country.copy()
    florida = florida[['cdc_case_earliest_dt', 'race_ethnicity_combined',
                       'Gender', 'Hospitalized',  'Died', 'res_state', 'Age_group']]
    return florida



def get_datafiles_ready():
    florida = pd.read_csv(os.path.join(CLEANDIR, f'florida_{FLORIDA_DATE}.csv'),index_col=False, parse_dates=['Case_', "ChartDate"])
    cdc = pd.read_csv(os.path.join(CLEANDIR, f"cdc_{CDC_DATE}_total.csv"),index_col=False, parse_dates=['cdc_case_earliest_dt'])



#%%
#CLEANDIR = "../../cleaned_data"

#florida = get_florida_data_ready()
#florida.to_csv(os.path.join(CLEANDIR, f'florida_{FLORIDA_DATE}.csv'), index=False)
#
#florida = pd.read_csv(os.path.join(CLEANDIR, f'florida_{FLORIDA_DATE}.csv'),index_col=False, parse_dates=['Case_', "ChartDate"])
#


#%%


cdc = get_cdc_data_ready()

cdc.to_csv(os.path.join(CLEANDIR, f"cdc_{CDC_DATE}_total.csv"))

#%%
import pandas as pd
cdc = pd.read_csv(os.path.join(CLEANDIR, f"cdc_{CDC_DATE}_total.csv"),index_col=False, parse_dates=['cdc_case_earliest_dt'])




#%% 
