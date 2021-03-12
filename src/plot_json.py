import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Output, Input
import plotly.graph_objs as go


cache = "fig.json"
# Construct a figure object and save it as json.
# df = px.data.iris()
# fig = px.scatter(df, x="sepal_width", y="sepal_length")

fig= go.Figure()
fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
    y=[0, 4, 5, 1, 2, 3, 2, 4, 2, 1],
    mode="text",
    text=["","","","", "No data.", "","","", "", ''],
    textfont_size=40,
    ))

fig.update_layout(
    font = dict(size=16),
    margin={'l':0, 'r':0, 't':0, 'b':0},
    width = 480,
    height=300
)

#fig= go.Figure()


with open(cache, 'w') as f:

    f.write(fig.to_json())
# Create example app.
app = dash.Dash(prevent_initial_callbacks=True)
app.layout = html.Div([dcc.Graph(id="graph"), html.Button("Click me", id="btn")])


@app.callback(Output("graph", "figure"), [Input("btn", "n_clicks")])
def func(n_clicks):
    #with open(cache, 'r') as f:
    return fig

if __name__ == '__main__':
    app.run_server()