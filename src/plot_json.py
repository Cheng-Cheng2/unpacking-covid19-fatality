import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Output, Input

cache = "fig.json"
# Construct a figure object and save it as json.
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length")
with open(cache, 'w') as f:
    f.write(fig.to_json())
# Create example app.
app = dash.Dash(prevent_initial_callbacks=True)
app.layout = html.Div([dcc.Graph(id="graph"), html.Button("Click me", id="btn")])


@app.callback(Output("graph", "figure"), [Input("btn", "n_clicks")])
def func(n_clicks):
    with open(cache, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    app.run_server()