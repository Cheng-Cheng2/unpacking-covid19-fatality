# %%

import pandas as pd

vrs = ['cdc_case_earliest_dt', 'race_ethnicity_combined', 'Gender', 'Hospitalized',  'Died', 'res_state', 'Age_group']
data_dir = '../../data'
#country = pd.read_csv(os.path.join(data_dir, 'COVID_Cases_Restricted_Detailed_12312020.csv'), parse_dates=['cdc_report
fname = 'COVID_Cases_Restricted_Detailed_02282021_Part_1.csv'
country = pd.read_csv(os.path.join(data_dir, fname), parse_dates=['cdc_report_dt', 'pos_spec_dt', 'onset_dt', 'cdc_case_earliest_dt'])
# %%
fname = 'COVID_Cases_Restricted_Detailed_02282021_Part_2.csv'
country2 = pd.read_csv(os.path.join(data_dir, fname), parse_dates=['cdc_report_dt', 'pos_spec_dt', 'onset_dt', 'cdc_case_earliest_dt'])


# %% 
fname =  'COVID_Cases_Restricted_Detailed_02282021_Part_3.csv'
country3 = pd.read_csv(os.path.join(data_dir, fname), parse_dates=['cdc_report_dt', 'pos_spec_dt', 'onset_dt', 'cdc_case_earliest_dt'])

# %%
print(country.shape, country2.shape, country3.shape)


# %%
vrs = ['current_status', 'cdc_case_earliest_dt', 'race_ethnicity_combined', 'sex', 'hosp_yn',  'death_yn', 'res_state', 'age_group']

country = country[vrs]
country2 = country2[vrs]
country3 = country3[vrs]

total = pd.concat([country, country2, country3], ignore_index=True)




# %%
import os 
total.to_csv(os.path.join(data_dir, 'CDC_02282021_total.csv'), index=False)
# %%
