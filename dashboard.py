from datetime import date
import dash # v 1.16.2
import dash_core_components as dcc # v 1.12.1
import dash_bootstrap_components as dbc # v 0.10.3
import dash_html_components as html # v 1.1.1
import pandas as pd
import plotly.express as px # plotly v 4.7.1
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Interactive Covid-19 Dashboard', external_stylesheets=[external_stylesheets])

df = pd.read_csv('Data/owid-covid-data.csv') #LOAD DATASET

df['continent'] = df['continent'].replace(np.nan, 'World', regex=True) #REPLACE LABELS WITHOUT CONTINENT

#replace new_deaths empty with the mean of that date
new_cases0 = np.where(pd.isnull(df['new_deaths']))[0]
for i in new_cases0:
    date_query = str(df['date'].loc[i])
    query = df.query("date == @date_query")
    df['new_deaths'].loc[i] = query['new_deaths'].mean()

new_cases0 = np.where(pd.isnull(df['new_deaths']))[0] # Remove rows from 2020-01-01 to 2020-01-21
df = df.drop(new_cases0)
df = df.drop(np.where(pd.isnull(df['population']))[0]) # Remove international rows
# REMOVE NaN replace it with 0
df['total_deaths'].replace(np.nan, 0, regex=True)
df['total_cases'].replace(np.nan, 0, regex=True)
df['total_cases_per_million'].replace(np.nan, 0, regex=True)
df['total_deaths_per_million'].replace(np.nan, 0, regex=True)
df['gdp_per_capita'].replace(np.nan, 0, regex=True)
df['population_density'].replace(np.nan, 0, regex=True)
df['life_expectancy'].replace(np.nan, 0, regex=True)
df['median_age'].replace(np.nan,1,regex = True)
# Not in Use
points_size_features = ['population', 'gdp_per_capita', 'population_density', 'life_expectancy']
features = ['new_deaths', 'new_cases', 'total_cases', 'total_deaths',
          'new_deaths_per_million','new_cases_per_million','total_cases_per_million',
         'total_deaths_per_million']

# Set the mean per date of the new_cases

date_group = df.groupby(['date'])
mean_new_cases =date_group['new_cases'].mean()

df_average = df[features].mean()
max_val = df[features].max().max()

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Label('Date Selection'),], style={'font-size': '18px'}),
            
           dcc.DatePickerSingle(
                id='my-date-picker-single',
                display_format='Y-MM-DD',
                month_format='MMMM Y',
                placeholder='MMMM Y',
                min_date_allowed=date(2020, 1, 23),
                max_date_allowed=date(2020, 12, 4),
                initial_visible_month=date(2020, 6, 1),
                date=date(2020, 3, 30)
            )  ,
            html.Div(id='output-container-date-picker-range'),
        ], style={'margin-left': '30px'}),
        html.Div([
            
            html.Div([
                html.Label('Feature Selection'),], style={'font-size': '18px'}),
            
            dcc.Dropdown(
                id='crossfilter-model',
                options=[

                    {'label':'Nuevas Muertes','value': 'new_deaths'},
                    {'label':'Nuevos Casos', 'value': 'new_cases'},
                    {'label':'Total Muertes','value': 'total_deaths'},
                    {'label':'Total Casos', 'value': 'total_cases'},
                    {'label':'Total Muertes por Millón','value': 'total_deaths_per_million'},
                    {'label':'Total Casos por Millón', 'value': 'total_cases_per_million'},
                    
                ],
                value = 'new_deaths',
                clearable = False

            )], style={'width': '49%', 'display': 'inline-block'}
        ),

        html.Div([
            
            html.Div([
                html.Label('Feature size selection'),], style={'font-size': '18px', 'width': '40%', 'display': 'inline-block'}),
            
            
            
            dcc.Dropdown(
                id='crossfilter-feature',
                #options=[{'label': i, 'value': i} for i in  ['None','Region','Channel','Total_Spend'] + features ],
                 options=[
                    {'label':'Población','value': 'population'},
                    {'label':'GDP per Capita','value': 'gdp_per_capita'},
                    {'label':'Densidad de Población','value': 'population_density'},
                    {'label':'Expectativa de Vida','value': 'life_expectancy'},
                     
                 ],
                value='population',
                clearable=False
            )], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
        
        )], style={'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px 5px'}
    ),

    html.Div([

        dcc.Graph(
            id='scatter-plot',
            hoverData={'points': [{'hovertext': 0}]}
        )

    ], style={'width': '100%', 'height':'90%', 'display': 'inline-block', 'padding': '0 20'}),
    
    
    html.Div([
                dcc.RadioItems(
                    id='gradient-scheme',
                    options=[
                        {'label': 'Orange to Red', 'value': 'OrRd'},
                        {'label': 'Viridis', 'value': 'Viridis'},
                        {'label': 'Plasma', 'value': 'Plasma'},
                    ],
                    value='Plasma',
                    labelStyle={'float': 'right', 'display': 'inline-block', 'margin-right': 10}
                ),
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
    
    
    html.Div([
         dcc.Graph(id='point-plot'),
    ], style={'display': 'inline-block', 'width': '100%'}),
    
     html.Div([
         dcc.Graph(id='pie-chart'),
    ], style={'display': 'inline-block', 'width': '100%'}),
    
    
    
    html.Div([
        dcc.RadioItems(
            id='case_value', 
            options=[
                        {'value': 'new_cases', 'label': 'Nuevos casos'} ,
                        {'value': 'new_deaths', 'label': 'Nuevas muertes'} ,
                     
                    ],
            value='new_cases',
            labelStyle={'float': 'right', 'display': 'inline-block', 'margin-right': 10}
        ),
        dcc.Graph(id="choropleth"),
    ],style={'width': '50%', 'display': 'inline-block', 'float': 'right'}),
    
    html.Div([
        dcc.Graph(id="choropleth2"),
    ],style={'width': '50%', 'display': 'inline-block', 'float': 'left'}),
    
    ], style={'backgroundColor': 'rgb(17, 17, 17)'},
)



@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    [
        dash.dependencies.Input('crossfilter-feature','value'),
        dash.dependencies.Input('crossfilter-model','value'),
        dash.dependencies.Input('my-date-picker-single', 'date')
    ]
)

def update_graph(feature, model, date_value):
    string_prefix = 'You have selected: '
    if date_value is not None:
        fecha = date_value
    else:
        fecha = '2020-03-30'
        
    
    specific_day = df.query("date == @fecha")
    #specific_day = specific_day[specific_day.location != 'World']
    max_val = specific_day['population'].max()
    sizes = [np.max([max_val/2,val]) for val in specific_day['population'].values]
    
    fig = px.scatter(
        specific_day,
        x = specific_day[feature],
        y = specific_day[model],
        opacity = 0.8,
        template = 'plotly_dark',
        hover_name = specific_day['location'],
        color = specific_day['continent'],
        size = sizes,
    )
    
    fig.update_traces(customdata = specific_day.index)
    
    fig.update_layout(
        title=feature.upper() + ' Vs ' + model.upper() +  ' - Date [ ' +  fecha + ' ]',
        height = 450,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 30},
        hovermode =  'closest',
        template = 'plotly_dark',
        
    )
   
    fig.update_xaxes(showticklabels=False, type='log')
    fig.update_yaxes(showticklabels=False, type='log')
    
    return fig


def create_point_plot(location,gradient):
    country_data = df.query("location == @location")
    fig = px.bar(
        country_data, 
        x = 'date', 
        y='new_cases',
        hover_data = ['total_deaths','date'],
        color = 'new_deaths',
        color_continuous_scale = gradient,
        labels={'new_cases': 'New Cases', 'date': 'Date',
                'new_deaths': 'New Deaths', 'total_deaths': 'Total Deaths'}, 
        title=' ' + location + ' Covid Historical'
    )
    fig.add_trace(
        go.Scatter(x=mean_new_cases.index,y=mean_new_cases,
                    name='Average')
    )
    fig.update_layout(
        barmode='group',
        height=280,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 30},
        template='plotly_dark',
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=False)
    
    fig.update_yaxes( type='log', range=[0,5])
    return fig


@app.callback(
    dash.dependencies.Output('point-plot','figure'),
    [
        dash.dependencies.Input('scatter-plot','hoverData'),
        dash.dependencies.Input('gradient-scheme','value'),
    ])
def update_point_plot(hoverData,gradient):
    location = hoverData['points'][0]['hovertext']
    title = f'Country {location}'
    return create_point_plot(str(location),gradient)


def create_pie_chart(location):
    
    if location == '0':
        country = "Colombia"
    else:
        country = location
    country_data = df.query("location == @country")
    
    
    
    labels = ['65 - 69 Older','70 Older']
    median_age = country_data['median_age'].mean()
    
    values = [country_data['aged_70_older'].mean(),country_data['aged_65_older'].mean()]
    
    continent = country_data['continent'].unique()[0]
    world_data = df.groupby(['continent']).get_group((continent))
    continent_data = world_data.groupby(['location'])['gdp_per_capita']
    values_countries = continent_data.mean()
    labels_countries = list(values_countries.index)
    pull = np.zeros([1,len(labels_countries)])
    index = labels_countries.index(country)
    pull[0,index] = 0.2
    
    
    colors = ['#5f007f','#2335be']
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]])
    
    fig.add_trace(go.Pie(
        values=values,
        labels=labels,
        domain=dict(x=[0, 0.1]),
        name= country + " Over 65 years infected",
        marker=dict(colors=colors)),
        row=1, col=1)
    
    fig.add_trace(go.Pie(
        labels=labels_countries, 
        values=values_countries, 
        sort=False, 
        pull=pull[0],
        domain=dict(x=[0.1, 1]),
        name= "GDP per Capita"),
        row=1, col=2)
    
    fig.update_layout(
        
        height = 550,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 150},
        hovermode =  'closest',
        template = 'plotly_dark',
        title=country + " Over 65 years infected - Median Age [ " + str(median_age) + " ]" + ' || '+ " GDP Continent Comparison")
   
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig
    
@app.callback(
    dash.dependencies.Output('pie-chart','figure'),
    [
        dash.dependencies.Input('scatter-plot','hoverData'),
    ])

def update_pie_chart(hoverData):
    location = hoverData['points'][0]['hovertext']
    return create_pie_chart(str(location))

@app.callback(
    dash.dependencies.Output("choropleth", "figure"), 
    [
        dash.dependencies.Input("case_value", "value"),
        dash.dependencies.Input('gradient-scheme','value'),
    ])
def display_choropleth(value,gradient):
    
    df_new = df[df.location != 'World']

    fig = px.choropleth(
        df_new, 
        locations="iso_code",
        color=value, 
        hover_name="location",
        animation_frame='date',
        color_continuous_scale=gradient)
    
    fig.update_geos(fitbounds="locations", visible=False)
    
    fig.update_layout(
        
        height = 550,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 150},
        hovermode =  'closest',
        template = 'plotly_dark',
        title=value + ' Animated representation',
        transition = {'duration': 10000})
    
    
    return fig

@app.callback(
    dash.dependencies.Output("choropleth2", "figure"), 
    [
        dash.dependencies.Input("case_value", "value")
    ])

def display_choropleth2(value):
    
    max_val = df["new_deaths"].max()

    sizes = [np.max([max_val/2,val]) for val in df["new_deaths"].values] 

    fig = px.scatter(
        df, 
        x='population', 
        y = value, 
        color='continent',
        hover_name='location',
        animation_frame='date', 
        animation_group='location',
        hover_data = ['new_cases','new_deaths'], 
        size= sizes, 
        log_y = True, 
        log_x = True)
    
    fig.update_yaxes( type='log', range=[-2,7])
    fig.update_xaxes( type='log', range=[5,11])
    
    fig.update_layout(
        
        height = 550,
        margin={'l': 20, 'b': 30, 'r': 10, 't': 150},
        hovermode =  'closest',
        template = 'plotly_dark',
        title=value + ' points representation',
        transition = {'duration': 1000})
    
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
