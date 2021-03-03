from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt
import dash
import dash_html_components as html
import dash_core_components as dcc
from plotting_utils import *
from dash.dependencies import Input, Output



app= dash.Dash()
# prepare data and plot 
florida = get_florida_data_ready()
#fig= plt.figure()
#ax= fig.add_subplot(111)
#ax.plot(range(10), [i**2 for i in range(10)])
#ax.grid(True)

app.layout= html.Div([
    # dcc.Checklist(
    #     id = "gender-checklist",
    #     options=[
    #         {'label': 'All', 'value': 'All'},
    #         {'label': 'Female', 'value': 'Female'},
    #         {'label': 'Male', 'value': 'Male'}
            
    #     ],
    #     value=['All'],
    #     labelStyle={'display': 'inline-block'}

    # ) ,

    dcc.Dropdown(
        id='gender-dropdown',
         options = [{'label': 'All', 'value': 'All'},
            {'label': 'Female', 'value': 'Female'},
            {'label': 'Male', 'value': 'Male'}
            
        ],
        value='All',
    ),
    #dcc.Graph(id= 'cases-graph', figure=plotly_fig)
    dcc.Graph(id= 'cases-graph')

])


@app.callback(Output(component_id='cases-graph', component_property='figure'),
                Input(component_id='gender-dropdown', component_property='value'))
def update_data(input_value):
    fig = florida_case_hosp_death(florida, input_value)
    plotly_fig = mpl_to_plotly(fig)
    return plotly_fig
 # return [ fig_1 , fig_2]


app.run_server(debug=True, port=8010, host='localhost')