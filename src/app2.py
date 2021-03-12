from datetime import date
from os.path import split

from pandas.io.formats import style
from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt
import dash
import dash_html_components as html
import dash_core_components as dcc
from plotting_utils import *
from dash.dependencies import Input, Output
import numpy as np
#import dash_html_components as html
from hfr_for_viz import compute_hfr_estimates
import dash_table
#from dash_extensions.enrich import Dash, ServersideOutput, Output, Input, Trigger


app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://codepen.io/chriddyp/pen/bWLwgP.css'
    ]
)
# prepare data and plot
florida, cdc = get_datafiles_ready()

race_cats = ['All']+cdc['race_ethnicity_combined'].dropna().unique().tolist()
race_cats = [x for x in race_cats if (x != 'Missing')  and (x!='Unknown')]
states_cats = cdc['res_state'].dropna().unique()#.tolist()
states_cats = ['All']+[x for x in states_cats if (x != 'Missing') and (x!= 'Unknown')]
app.layout = html.Div([
    dcc.Store(id="store"),  # this is the store that holds the data
    html.Div(id="onload"),  # this div is used to trigger the query_df function on page load

    html.H1(children="Unpacking the Drop in COVID-19 CFR: A Study of National and Florida Line-Level Data"),
    # https://dash.plotly.com/sharing-data-between-callbacks
    #html.Button('Load data', id='load-data-button', n_clicks=0),
    # dcc.Loading(
    #     id="loading-data",
    #     type='default',
    #     children = html.Div(id='load-data')
    # ),

    html.H2(children="COVID-19 Cases, Hospitalizations, and Deaths", style={'backgroundColor':'yellow'}),
   
    # html.H6(children="Gender"),
    html.H2(children="Aggregate", style={'backgroundColor':'grey'}),    

    html.Div([
        html.Div([
            html.H6('Gender (for both Florida and national)'),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[{'label': 'All', 'value': 'All'},
                         {'label': 'Female', 'value': 'Female'},
                         {'label': 'Male', 'value': 'Male'}

                         ],
                value='All',
            ),
        ], className="four columns"),

        html.Div([
            #html.H4("For country only:", style={"background-color": "grey"}),
            html.H6('Race (for national only)'),
            dcc.Dropdown(
                id='race-dropdown',
                options=[{'label': i, 'value': i} for i in race_cats],
                value='All',
            ),
        ], className="four columns"),

        html.Div([
            #html.H4("\n"),
            html.H6('State (for national only)'),
            dcc.Dropdown(
                id='state-dropdown',
                options=[{'label': i, 'value': i} for i in states_cats],
                value='All',
            ),
        ], className="four columns"),

    ],
        className="row"
    ),
    
    
    # aggregate FLorida cases, hosps,deaths
     # Florida cases, hosps, deaths
   
    html.H3(children="Florida"),
    html.Div([
        html.Div([
            html.H3('Cases'),
            dcc.Graph(id='agg-f-case-g')
        ], className="four columns"),

        html.Div([
            html.H3('Hospitalizations'),
            dcc.Graph(id='agg-f-hosp-g')
        ], className="four columns"),

        html.Div([
            html.H3('Deaths'),
            dcc.Graph(id='agg-f-died-g')
        ], className="four columns"),
    ],
        className="row"
    ),
    html.H3(children='National'),
    html.Div([
        html.Div([
            html.H3('Cases'),
            dcc.Graph(id='agg-n-case-g')
        ], className="four columns"),

        html.Div([
            html.H3('Hospitalizations'),
            dcc.Graph(id='agg-n-hosp-g')
        ], className="four columns"),

        html.Div([
            html.H3('Deaths'),
            dcc.Graph(id='agg-n-died-g')
        ], className="four columns"),
    ],
        className="row"
    ),
html.H6("For patient privacy protection, less or equal to 5 counts of cases, hospitalizations, and deaths on any given day would not be plotted.", style={'color':'blue'}),
    # Age separated Florida cases, hosps, deaths
    html.H2(children="Age-stratified", style={'backgroundColor':'grey'}),    
     html.Div([
        html.Div([
            html.H6('Gender (for both Florida and national)'),
            dcc.Dropdown(
                id='gender-dropdown-str',
                options=[{'label': 'All', 'value': 'All'},
                         {'label': 'Female', 'value': 'Female'},
                         {'label': 'Male', 'value': 'Male'}

                         ],
                value='All',
            ),
        ], className="four columns"),

        html.Div([
            #html.H4("For country only:", style={"background-color": "grey"}),
            html.H6('Race (for national only)'),
            dcc.Dropdown(
                id='race-dropdown-str',
                options=[{'label': i, 'value': i} for i in race_cats],
                value='All',
            ),
        ], className="four columns"),

        html.Div([
            #html.H4("\n"),
            html.H6('State (for national only)'),
            dcc.Dropdown(
                id='state-dropdown-str',
                options=[{'label': i, 'value': i} for i in states_cats],
                value='All',
            ),
        ], className="four columns"),

    ],
        className="row"
    ),
    html.H3(children="Florida"),
    html.Div([
        html.Div([
            html.H3('Cases'),
            dcc.Graph(id='f-case-g')
        ], className="four columns"),

        html.Div([
            html.H3('Hospitalizations'),
            dcc.Graph(id='f-hosp-g')
        ], className="four columns"),

        html.Div([
            html.H3('Deaths'),
            dcc.Graph(id='f-died-g')
        ], className="four columns"),
    ],
        className="row"
    ),
    html.H3(children='National'),
    html.Div([
        html.Div([
            html.H3('Cases'),
            dcc.Graph(id='n-case-g')
        ], className="four columns"),

        html.Div([
            html.H3('Hospitalizations'),
            dcc.Graph(id='n-hosp-g')
        ], className="four columns"),

        html.Div([
            html.H3('Deaths'),
            dcc.Graph(id='n-died-g')
        ], className="four columns"),
    ],
        className="row"
    ),
    html.H6("For patient privacy protection, less or equal to 5 counts of cases, hospitalizations, and deaths on any given day would not be plotted.", style={'color':'blue'}),
    # age distribution section
    html.H2(children="COVID-19 Age Distributions among Cases, Hospitalizations, and Deaths", style={'backgroundColor':'yellow'}),
    html.Div([
        html.Div([
            html.H6('Gender (for both Florida and national)'),
            dcc.Dropdown(
                id='age-gender-dropdown',
                options=[{'label': 'All', 'value': 'All'},
                         {'label': 'Female', 'value': 'Female'},
                         {'label': 'Male', 'value': 'Male'}

                         ],
                value='All',
            ),
        ], className="four columns"),

        html.Div([
            #html.H4("For country only:", style={"background-color": "grey"}),
            html.H6('Race (for national only)'),
            dcc.Dropdown(
                id='age-race-dropdown',
                options=[{'label': i, 'value': i} for i in race_cats],
                value='All',
            ),
        ], className="four columns"),

        html.Div([
            #html.H4("\n"),
            html.H6('State (for national only)'),
            dcc.Dropdown(
                id='age-state-dropdown',
                options=[{'label': i, 'value': i} for i in states_cats],
                value='All',
            ),
        ], className="four columns"),

    ],
        className="row"
    ),
    
    
    html.H3(children="Florida"),
    html.Div([
        html.Div([
            html.H3('Cases'),
            dcc.Graph(id='age-f-case-g')
        ], className="four columns"),

        html.Div([
            html.H3('Hospitalizations'),
            dcc.Graph(id='age-f-hosp-g')
        ], className="four columns"),

        html.Div([
            html.H3('Deaths'),
            dcc.Graph(id='age-f-died-g')
        ], className="four columns"),
    ],
        className="row"
    ),
    html.H3(children='National'),
    html.Div([
        html.Div([
            html.H3('Cases'),
            dcc.Graph(id='age-n-case-g')
        ], className="four columns"),

        html.Div([
            html.H3('Hospitalizations'),
            dcc.Graph(id='age-n-hosp-g')
        ], className="four columns"),

        html.Div([
            html.H3('Deaths'),
            dcc.Graph(id='age-n-died-g')
        ], className="four columns"),
    ],
        className="row"
    ),
    


 # HFR estimates nationally and for florida
    html.H2(children="Age-stratified HFR Estimates", style={'backgroundColor':'yellow'}),

    ###########DATES
    html.H6("Choose two dates for estimating HFR drops"),
    html.H5("**WARNING: after 2020-12-01, due to ramp-ups of vaccines, HFR estimates might not reflect all treatment improvements.**", style={'color':'red'}),
    html.Div([
        html.Div([
            #html.H3('Florida'),
            #dcc.Input(id="date1", type="text", placeholder="2020-04-01")
            dcc.DatePickerRange(
                id='date1',
                min_date_allowed="2020-04-01",
                max_date_allowed="2021-02-01",
                initial_visible_month="2020-04-01",
                start_date = '2020-04-01',
                end_date="2020-12-01"
            ),
        ], className="six columns"),
    ],
        className="row"
    ),

    ###########tables
    html.Div([
        html.Div([
            html.H6('Gender (for both Florida and national)'),
            dcc.Dropdown(
                id='hfr-gender-dropdown',
                options=[{'label': 'All', 'value': 'All'},
                         {'label': 'Female', 'value': 'Female'},
                         {'label': 'Male', 'value': 'Male'}

                         ],
                value='All',
            ),
        ], className="four columns"),

        html.Div([
            #html.H4("For country only:", style={"background-color": "grey"}),
            html.H6('Race (for national only)'),
            dcc.Dropdown(
                id='hfr-race-dropdown',
                options=[{'label': i, 'value': i} for i in race_cats],
                value='All',
            ),
        ], className="four columns"),

        html.Div([
            #html.H4("\n"),
            html.H6('State (for national only)'),
            dcc.Dropdown(
                id='hfr-state-dropdown',
                options=[{'label': i, 'value': i} for i in states_cats],
                value='All',
            ),
        ], className="four columns"),

    ],
        className="row"
    ),
    
    html.Div([
        html.Div([
            html.H3('Florida'),
            html.Div(id='florida-table')
        ], className="six columns"),

        html.Div([
            html.H3('National'),
            html.Div(id='national-table')
        ], className="six columns"),
    ],
        className="row"
    ),

    html.H5("**Note: we ommit age groups where less than two deaths occured for reliability.**", style={'color':'red'}),

],
    style={'marginLeft': 5, 'marginRight': 20}
)
# app.css.append_css({
#     "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
# })

"""  @app.callback(Output('load-data', 'children'), Input(''))
def get_data(value): 
    florida, cdc = get_datafiles_ready()
    return [florida.to_json(date_format='iso', orient='split'), cdc.to_json(date_format='iso', orient='split')] 
 """
# loading data 



#update aggregate cases, hosptitalizatons, deaths

@app.callback(Output(component_id='agg-f-case-g', component_property='figure'),
              Output(component_id='agg-f-hosp-g', component_property='figure'),
              Output(component_id='agg-f-died-g', component_property='figure'),
              Input(component_id='gender-dropdown', component_property='value'))
def update_count_figures(input_value):
    figs = florida_case_hosp_death_agg(florida, input_value)
    return [figs[0], figs[1], figs[2]]
@app.callback(Output(component_id='agg-n-case-g', component_property='figure'),
              Output(component_id='agg-n-hosp-g', component_property='figure'),
              Output(component_id='agg-n-died-g', component_property='figure'),
              Input(component_id='gender-dropdown', component_property='value'),
              Input(component_id='race-dropdown', component_property='value'),
              Input(component_id='state-dropdown', component_property='value'))
def update_count_figures_national(gender, race, state):
    figs = florida_case_hosp_death_agg(cdc, gender, 'cdc_case_earliest_dt', race, state)
    return [figs[0], figs[1], figs[2]]


# update age seperated cases, hosps, deaths
@app.callback(Output(component_id='f-case-g', component_property='figure'),
              Output(component_id='f-hosp-g', component_property='figure'),
              Output(component_id='f-died-g', component_property='figure'),
              Input(component_id='gender-dropdown-str', component_property='value'))
def update_count_figures(input_value):
    figs = florida_case_hosp_death(florida, input_value)
    return [figs[0], figs[1], figs[2]]
@app.callback(Output(component_id='n-case-g', component_property='figure'),
              Output(component_id='n-hosp-g', component_property='figure'),
              Output(component_id='n-died-g', component_property='figure'),
              Input(component_id='gender-dropdown-str', component_property='value'),
              Input(component_id='race-dropdown-str', component_property='value'),
              Input(component_id='state-dropdown-str', component_property='value'))
def update_count_figures_national(gender, race, state):
    figs = florida_case_hosp_death(cdc, gender, 'cdc_case_earliest_dt', race, state)
    return [figs[0], figs[1], figs[2]]

# update age distrubution plots
@app.callback(Output(component_id='age-f-case-g', component_property='figure'),
              Output(component_id='age-f-hosp-g', component_property='figure'),
              Output(component_id='age-f-died-g', component_property='figure'),
              Input(component_id='age-gender-dropdown', component_property='value'))
def update_age_figures(input_value):
    figs = age_distribution_plots(florida, input_value)
    return [figs[0], figs[1], figs[2]]

@app.callback(Output(component_id='age-n-case-g', component_property='figure'),
              Output(component_id='age-n-hosp-g', component_property='figure'),
              Output(component_id='age-n-died-g', component_property='figure'),
              Input(component_id='age-gender-dropdown', component_property='value'),
              Input(component_id='age-race-dropdown', component_property='value'),
              Input(component_id='age-state-dropdown', component_property='value'))
def update_age_figures_national(gender, race, state):
    figs = age_distribution_plots(cdc, gender, 'cdc_case_earliest_dt', race, state)
    return [figs[0], figs[1], figs[2]]




## florida national tables
@app.callback(Output(component_id='florida-table', component_property='children'),
              #Output(component_id='age-f-hosp-g', component_property='figure'),
              #Output(component_id='age-f-died-g', component_property='figure'),
              Input(component_id='date1', component_property='start_date'),
              Input(component_id='date1', component_property='end_date'),
              Input(component_id='hfr-gender-dropdown', component_property='value'))
def update_hfr_florida(date1, date2, input_value):
    if input_value == 'All':
        filters = None
    else:
        filters = {'Gender':input_value}
    print("**********************Dates: ", [date1, date2])




    df = compute_hfr_estimates(None, florida, [date1, date2], filters, update_florida=True)


    if df is None:
        return html.Div(
            html.H6("Not enough support", style={'color':'blue'})
        )
    df = df.reset_index().rename(columns={'index': ''})
    return html.Div(
        [
            dash_table.DataTable(
                data=df.to_dict("rows"),
                columns=[{"id": x, "name": x} for x in df.columns],
            )
        ]
    )

@app.callback(Output(component_id='national-table', component_property='children'),
              #Output(component_id='age-n-hosp-g', component_property='figure'),
              #Output(component_id='age-n-died-g', component_property='figure'),
              Input(component_id='date1', component_property='start_date'),
              Input(component_id='date1', component_property='end_date'),
              Input(component_id='hfr-gender-dropdown', component_property='value'),
              Input(component_id='hfr-race-dropdown', component_property='value'),
              Input(component_id='hfr-state-dropdown', component_property='value'))
def update_hfr_national(date1, date2, gender='All', race='All', state='All'):
    filters = {}
    if gender=='All' and race=='All' and state=='All':
        filters = None
    else:
        if gender != 'All':
            filters['Gender'] = gender
        if race != 'All':
            filters['race_ethnicity_combined'] = race
        if state != 'All':
            filters['res_state'] =state
    df = compute_hfr_estimates(cdc, None, [date1, date2], filters, update_florida=False)    
    if df is None:
        return html.Div(
            html.H6("Not enough support", style={'color':'blue'})
        )

    df = df.reset_index().rename(columns={'index': ''})


    return html.Div(
        [
            dash_table.DataTable(
                data=df.to_dict("rows"),
                columns=[{"id": x, "name": x} for x in df.columns],
            )
        ]
    )



if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8010)



