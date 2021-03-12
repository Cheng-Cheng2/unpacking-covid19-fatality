import dash
import dash_html_components as html
import dash_core_components as dcc



import dash
import dash_html_components as html
import dash_core_components as dcc


app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://codepen.io/chriddyp/pen/bWLwgP.css'
    ]
)


app.layout = html.Div([
    html.H1("Appendix")
])



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8015)

