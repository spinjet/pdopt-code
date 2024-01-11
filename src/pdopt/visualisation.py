# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:55:50 2021

@author: s345001
"""

# Standard Library Imports
import base64
import io
import re
from functools import reduce


# Third-party imports
import numpy as np
import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

import plotly.io as pio

pio.renderers.default = 'browser'

# Local imports

# Module Constants
__author__ = 'Andrea Spinelli'
__copyright__ = 'Copyright 2021, all rights reserved'
__status__ = 'Development'

pd.options.mode.chained_assignment = None

NONE_LABELS = [{'label': 'None', 'value': 'none'}]

LOAD_MODULE = html.Div(children=[
    dcc.Upload(id='upload-data',
               children=[
                   html.A('Load .csv file')],
               style={
                   'width': 'fit-content',
                   'height': '60px',
                   'lineHeight': '60px',
                   'borderWidth': '1px',
                   'borderStyle': 'dashed',
                   'borderRadius': '5px',
                   'textAlign': 'center',
                                'margin': '10px',
                                'float': 'left',
                                'display': 'inline-block',
                                'padding': '1px 10px',
               },)
])


# Scatter Sizes
SCATTER_SIZES = ['45vw', '45vw', '31vw', '31vw', '31vw']

# Color Schemes
SCATTER_COLOR = 'rainbow'
PARCOORD_COLOR = 'turbo'
PARCOORD_COLOR_LIST = [
    'rgb(48,18,59)',
    'rgb(65,69,171)',
    'rgb(70,117,237)',
    'rgb(57,162,252)',
    'rgb(27,207,212)',
    'rgb(36,236,166)',
    'rgb(97,252,108)',
    'rgb(164,252,59)',
    'rgb(209,232,52)',
    'rgb(243,198,58)',
    'rgb(254,155,45)',
    'rgb(243,99,21)',
    'rgb(217,56,6)',
    'rgb(177,25,1)',
    'rgb(122,4,2)']

STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Because dash is a piece of crap, I have to use globals or nothing works
g_app = dash.Dash(__name__, external_stylesheets=STYLESHEETS)

g_dataframe = None
g_labels = None
g_customdata = None

g_parallel_coord_figure = None
g_scatter_list = []

g_selected_points_sca, g_selected_points_parc = [], []

g_par_coord_selected_ranges = {}
g_scatter_selection_list = [{}, {}, {}, {}, {}]

# Module Functions


def get_HTML_Layout(ext_load=True, load_figures=False):
    global g_lables, g_parallel_coord_figure, g_scatter_list
    # Returns the HTML structure of the app
    return html.Div([

        # Title
        html.H1(children='P-DOPT Visualisation Tool'),
        html.H5(children='This webapp allows to visualize .csv outputs from P-DOPT'),

        html.Div(id='debug'),

        # Load-in data
        LOAD_MODULE if ext_load else html.Br(),

        html.Br(),

        # Parallel Coordinates
        html.Div(id='parcoord-div',
                 children=[

                     html.Div(children=[

                         html.Div(children=[
                             'Parallel Coordinates Color',

                             dcc.Dropdown(id='dropdown-parcoord',
                                          options=g_labels if load_figures else NONE_LABELS,
                                          value=g_labels[-1]['value'] if load_figures else NONE_LABELS[0]['value'],
                                          )],

                                  style={'width': '20vw',
                                         'vertical-align': 'bottom',
                                         'display': 'inline-block'}
                                  ),

                         # Button to reset the scatter plot in case something goes wrong
                         # html.Div(children=[
                         #          html.Button('Reset',
                         #                      id='reset',
                         #                      n_clicks=0
                         #                      ),

                         #          html.Div(
                         #                  [html.Button("Download Image", id="btn_image"),
                         #                   dcc.Download(id="download-image")])
                         #          ],

                         #          style={'float': 'right',
                         #                'vertical-align': 'bottom',
                         #                'display': 'inline-block'}),
                     ]),

                     dcc.Graph(id='parcoord-figure',
                               figure=g_parallel_coord_figure if load_figures else go.Figure(),
                               style={'height': '90vh',
                                      'margin': '10px'})
                 ],

                 style={'borderLeft': 'thin grey solid',
                        'borderRight': 'thin grey solid',
                        'borderTop': 'thin grey solid',
                        'borderBottom': 'thin grey solid',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'backgroundColor': 'rgb(250, 250, 250)',
                        'padding': '10px 10px',
                        }
                 ),

        html.Br(),

        html.Div(children=[

            # Scatter Plot 1
            html.Div(id='sd1', children=[

                html.Div([
                    dcc.Graph(id='scatter-figure1',
                              figure=g_scatter_list[0] if load_figures else go.Figure(
                              ),
                              style={'height': '47vw', 'width': '47vw'}
                              )
                ]),

                # Drop Down menus to selects axes and colors
                html.Div(children=[
                    html.Div(['X axis',

                              dcc.Dropdown(
                                  id='sd1-xaxis',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-1]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],
                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}
                             ),

                    html.Div(['Y axis',

                              dcc.Dropdown(
                                  id='sd1-yaxis',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-2]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],
                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}),

                    html.Div(['Color axis',

                              dcc.Dropdown(
                                  id='sd1-color',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-3]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],

                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}),
                ])
            ],

                style={
                'width': '47vw',
                'padding': '5px 5px',
                'display': 'inline-block',
                'float': 'left',
                'borderLeft': 'thin grey solid',
                'borderRight': 'thin grey solid',
                'borderTop': 'thin grey solid',
                'borderBottom': 'thin grey solid',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'backgroundColor': 'rgb(250, 250, 250)',
                'vertical-align': 'top',
            }
            ),

            # Scatter Plot 2
            html.Div(id='sd2', children=[

                html.Div([
                    dcc.Graph(id='scatter-figure2',
                              figure=g_scatter_list[1] if load_figures else go.Figure(
                              ),
                              style={'height': '47vw', 'width': '47vw'}
                              )
                ]),

                # Drop Down menus to selects axes and colors
                html.Div(children=[
                    html.Div(['X axis',

                              dcc.Dropdown(
                                  id='sd2-xaxis',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-1]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],
                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}
                             ),

                    html.Div(['Y axis',

                              dcc.Dropdown(
                                  id='sd2-yaxis',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-2]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],
                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}),

                    html.Div(['Color axis',

                              dcc.Dropdown(
                                  id='sd2-color',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-3]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],

                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}),
                ])
            ],

                style={
                'width': '47vw',
                'padding': '5px 5px',
                'display': 'inline-block',
                'float': 'right',
                'borderLeft': 'thin grey solid',
                'borderRight': 'thin grey solid',
                'borderTop': 'thin grey solid',
                'borderBottom': 'thin grey solid',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'backgroundColor': 'rgb(250, 250, 250)',
                'vertical-align': 'top',
            }),
        ]),

        html.Br(),

        html.Div(children=[

            # Scatter Plot 3
            html.Div(id='sd3', children=[
                 html.Div([
                     dcc.Graph(id='scatter-figure3',
                               figure=g_scatter_list[2] if load_figures else go.Figure(
                               ),
                               style={'height': '31vw', 'width': '31vw'}
                               )
                 ]),

                 # Drop Down menus to selects axes and colors
                 html.Div(children=[
                     html.Div(['X axis',

                          dcc.Dropdown(
                              id='sd3-xaxis',
                              options=g_labels if load_figures else NONE_LABELS,
                              value=g_labels[-1]['value'] if load_figures else NONE_LABELS[0]['value'],
                          ),
                     ],
                         style={'width': '30%',
                                'display': 'inline-block',
                                'padding': '5px'}
                     ),

                     html.Div(['Y axis',

                               dcc.Dropdown(
                                   id='sd3-yaxis',
                                   options=g_labels if load_figures else NONE_LABELS,
                                   value=g_labels[-2]['value'] if load_figures else NONE_LABELS[0]['value'],
                               ),
                               ],
                              style={'width': '30%',
                                     'display': 'inline-block',
                                     'padding': '5px'}),

                     html.Div(['Color axis',

                               dcc.Dropdown(
                                   id='sd3-color',
                                   options=g_labels if load_figures else NONE_LABELS,
                                   value=g_labels[-3]['value'] if load_figures else NONE_LABELS[0]['value'],
                               ),
                               ],

                              style={'width': '30%',
                                     'display': 'inline-block',
                                     'padding': '5px'}),
                 ])
                 ],

                style={
                'width': '31vw',
                'padding': '5px 5px',
                           'display': 'inline-block',
                           # 'float' : 'left',
                           'marginTop': '20px',
                           'marginRight': '15px',
                           'borderLeft': 'thin grey solid',
                           'borderRight': 'thin grey solid',
                           'borderTop': 'thin grey solid',
                           'borderBottom': 'thin grey solid',
                           'borderWidth': '1px',
                           'borderStyle': 'dashed',
                           'borderRadius': '5px',
                           'backgroundColor': 'rgb(250, 250, 250)',
                           'vertical-align': 'top',
            }),

            html.Div(id='sd4', children=[
                html.Div([
                     dcc.Graph(id='scatter-figure4',
                               figure=g_scatter_list[3] if load_figures else go.Figure(
                               ),
                               style={'height': '31vw', 'width': '31vw'}
                               )
                     ]),

                # Drop Down menus to selects axes and colors
                html.Div(children=[
                    html.Div(['X axis',

                              dcc.Dropdown(
                                  id='sd4-xaxis',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-1]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],
                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}
                             ),

                    html.Div(['Y axis',

                              dcc.Dropdown(
                                  id='sd4-yaxis',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-2]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],
                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}),

                    html.Div(['Color axis',

                              dcc.Dropdown(
                                  id='sd4-color',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-3]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],

                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}),
                ])
            ],

                style={
                'width': '31vw',
                'padding': '5px 5px',
                           'display': 'inline-block',
                           # 'float'  : 'center',
                           'marginTop': '20px',
                           'marginRight': '15px',
                           'borderLeft': 'thin grey solid',
                           'borderRight': 'thin grey solid',
                           'borderTop': 'thin grey solid',
                           'borderBottom': 'thin grey solid',
                           'borderWidth': '1px',
                           'borderStyle': 'dashed',
                           'borderRadius': '5px',
                           'backgroundColor': 'rgb(250, 250, 250)',
                           'vertical-align': 'top',
            }),

            html.Div(id='scat5', children=[
                html.Div([
                     dcc.Graph(id='scatter-figure5',
                               figure=g_scatter_list[4] if load_figures else go.Figure(
                               ),
                               style={'height': '31vw', 'width': '31vw'}
                               )
                     ]),

                # Drop Down menus to selects axes and colors
                html.Div(children=[
                    html.Div(['X axis',

                              dcc.Dropdown(
                                  id='sd5-xaxis',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-1]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],
                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}
                             ),

                    html.Div(['Y axis',

                              dcc.Dropdown(
                                  id='sd5-yaxis',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-2]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],
                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}),

                    html.Div(['Color axis',

                              dcc.Dropdown(
                                  id='sd5-color',
                                  options=g_labels if load_figures else NONE_LABELS,
                                  value=g_labels[-3]['value'] if load_figures else NONE_LABELS[0]['value'],
                              ),
                              ],

                             style={'width': '30%',
                                    'display': 'inline-block',
                                    'padding': '5px'}),
                ])
            ],

                style={
                'width': '31vw',
                'padding': '5px 5px',
                           'display': 'inline-block',
                           # 'float' : 'right',
                           'marginTop': '20px',
                           'borderLeft': 'thin grey solid',
                           'borderRight': 'thin grey solid',
                           'borderTop': 'thin grey solid',
                           'borderBottom': 'thin grey solid',
                           'borderWidth': '1px',
                           'borderStyle': 'dashed',
                           'borderRadius': '5px',
                           'backgroundColor': 'rgb(250, 250, 250)',
                           'vertical-align': 'top',
            }),

        ],
        ),

        html.Br(),
        html.Div(id='copyright', children=[
                 'V. 0.4, Copyright Andrea Spinelli']),

        # Local Variables
        dcc.Store(id='update_parc'),
        dcc.Store(id='update_scatter'),
        dcc.Store(id='isDataLoaded'),
    ])


def generate_parallel_coordinates(color_label):
    # Build Parallel Coordinates Graph Object
    global g_dataframe, g_labels, g_selected_points_parc

    df = g_dataframe.copy()

    if len(g_selected_points_parc) > 0:
        columns = list(df.columns)

        if color_label != 'none':

            df['dummy'] = df[color_label]
            index_unsel = [int(x) for x in df.index]

            # Remove points to keep
            for point in g_selected_points_parc:
                index_unsel.remove(point)

            for i in index_unsel:
                df['dummy'][i] = df[color_label].min() - 1

            k = (df[color_label].max() - (df[color_label].min() - 1))
            q = df[color_label].min() - 1
            color_values = np.linspace((df[color_label].min() - q) / k,
                                       (df[color_label].max() - q) / k,
                                       len(PARCOORD_COLOR_LIST))

            color_scale = [[color_values[i], PARCOORD_COLOR_LIST[i]]
                           for i in range(len(PARCOORD_COLOR_LIST))]
            color_scale = [[0, 'rgb(211,211,211)']] + color_scale

            parcoord = go.Parcoords(
                line=dict(color=df['dummy'],
                          colorscale=color_scale,
                          # alpha=0.6,
                          cauto=False,
                          cmin=df[color_label].min() - 1,
                          cmax=df[color_label].max(),
                          # colorbar={"title": color_label,
                          #           'tickmode' : 'array',
                          #           'tickvals': color_values,
                          #           }
                          ),

                dimensions=list([dict(range=[df[col].min(), df[col].max()],
                                      label='<b>*{}*</b>'.format(
                                          col) if col == color_label else col,
                                      values=df[col]) for col in columns])
            )

        else:

            # Dummy column with only two colors
            dummy = [
                1 if int(x) in g_selected_points_parc else 0 for x in df.index]
            df['dummy'] = dummy

            color_scale = [[0, 'rgb(211,211,211)'], [0.5, 'rgb(139,0,0)'], [
                1, 'rgb(139,0,0)']]

            parcoord = go.Parcoords(
                line=dict(color=df['dummy'],
                          # alpha=0.6,
                          colorscale=color_scale,
                          cauto=False,
                          cmin=0,
                          cmax=1,
                          # colorbar={"title": color_label,
                          #           'tickmode' : 'array',
                          #           'tickvals': color_values,
                          #           }
                          ),

                dimensions=list([dict(range=[df[col].min(), df[col].max()],
                                      label=col,
                                      values=df[col]) for col in columns])
            )

        fig_par = go.Figure(data=parcoord)

    else:
        columns = list(df.columns)

        # If color scale is not selected, make lines red
        if color_label != 'none':
            parcoord = go.Parcoords(
                line=dict(color=df[color_label],
                          colorscale=PARCOORD_COLOR_LIST,
                          cauto=True,
                          ),

                dimensions=list([dict(range=[df[col].min(), df[col].max()],
                                      label='<b>*{}*</b>'.format(
                                          col) if col == color_label else col,
                                      values=df[col]) for col in columns])
            )
        else:
            parcoord = go.Parcoords(
                line=dict(color='rgb(139,0,0)'),

                dimensions=list([dict(range=[df[col].min(), df[col].max()],
                                      label=col,
                                      values=df[col]) for col in columns])
            )

        fig_par = go.Figure(data=parcoord)

    return fig_par


def generate_scatter(x_label, y_label, color_label, scatter_id):
    # Build Scatter Plot Graph Object
    global g_dataframe, g_labels, g_selected_points_sca, g_scatter_selection_list, g_customdata

    if color_label != 'none':
        m = dict(size=10,
                 color=g_dataframe[color_label],
                 colorscale=SCATTER_COLOR,
                 showscale=True,
                 colorbar={"title": color_label,
                           'titleside': 'right'
                           },
                 line=dict(width=1, color='DarkSlateGrey')
                 )
    else:
        m = dict(size=10,
                 color='LightSkyBlue',
                 line=dict(width=1, color='DarkSlateGrey')
                 )

    scatter = go.Scatter(
        x=g_dataframe[x_label],
        y=g_dataframe[y_label],
        mode='markers',
        marker=m,
        customdata=g_customdata
    )

    fig_sca = go.Figure(data=scatter)
    fig_sca.update_layout(xaxis_title=x_label,
                          yaxis_title=y_label, hovermode='closest')

    if len(g_scatter_selection_list[scatter_id]) > 0:
        sel_range = g_scatter_selection_list[scatter_id]
        selection_bounds = {'x0': sel_range['x'][0], 'x1': sel_range['x'][1],
                            'y0': sel_range['y'][0], 'y1': sel_range['y'][1]}

        fig_sca.add_shape(dict({'type': 'rect',
                                'line': {'width': 1, 'dash': 'dot', 'color': 'darkgrey'}},
                               **selection_bounds))

    if len(g_selected_points_sca) > 0:
        fig_sca.update_traces(unselected={'marker': {'color': 'LightGray',
                                                     'opacity': 0.1}},
                              selectedpoints=g_selected_points_sca)

    # Add Hover Data
    fig_sca.update_traces(
        hovertemplate="<br>".join(
            ['{}:%{{customdata[{}]:.3f}}'.format(
                g_dataframe.columns[k], k) for k in range(len(g_dataframe.columns))]
        ))

    return fig_sca


@g_app.callback(Output('scatter-figure1', 'figure'),
                Input('sd1-xaxis', 'value'),
                Input('sd1-yaxis', 'value'),
                Input('sd1-color', 'value'),
                Input('update_scatter', 'data'),
                prevent_initial_call=True)
def update_scatter1(x_label, y_label, c_label, dummy):
    global g_dataframe, g_scatter_list
    g_scatter_list[0] = generate_scatter(x_label, y_label, c_label, 0)

    return g_scatter_list[0]


@g_app.callback(Output('scatter-figure2', 'figure'),
                Input('sd2-xaxis', 'value'),
                Input('sd2-yaxis', 'value'),
                Input('sd2-color', 'value'),
                Input('update_scatter', 'data'),
                prevent_initial_call=True)
def update_scatter2(x_label, y_label, c_label, dummy):
    global g_dataframe, g_scatter_list
    g_scatter_list[1] = generate_scatter(x_label, y_label, c_label, 1)

    return g_scatter_list[1]


@g_app.callback(Output('scatter-figure3', 'figure'),
                Input('sd3-xaxis', 'value'),
                Input('sd3-yaxis', 'value'),
                Input('sd3-color', 'value'),
                Input('update_scatter', 'data'),
                prevent_initial_call=True)
def update_scatter3(x_label, y_label, c_label, dummy):
    global g_dataframe, g_scatter_list
    g_scatter_list[2] = generate_scatter(x_label, y_label, c_label, 2)

    return g_scatter_list[2]


@g_app.callback(Output('scatter-figure4', 'figure'),
                Input('sd4-xaxis', 'value'),
                Input('sd4-yaxis', 'value'),
                Input('sd4-color', 'value'),
                Input('update_scatter', 'data'),
                prevent_initial_call=True)
def update_scatter4(x_label, y_label, c_label, dummy):
    global g_dataframe, g_scatter_list
    g_scatter_list[3] = generate_scatter(x_label, y_label, c_label, 3)

    return g_scatter_list[3]


@g_app.callback(Output('scatter-figure5', 'figure'),
                Input('sd5-xaxis', 'value'),
                Input('sd5-yaxis', 'value'),
                Input('sd5-color', 'value'),
                Input('update_scatter', 'data'),
                prevent_initial_call=True)
def update_scatter5(x_label, y_label, c_label, dummy):
    global g_dataframe, g_scatter_list, g_scatter_selection_list
    g_scatter_list[4] = generate_scatter(x_label, y_label, c_label, 4)

    return g_scatter_list[4]


@g_app.callback(Output('parcoord-figure', 'figure'),
                Input('dropdown-parcoord', 'value'),
                Input('update_parc', 'data'),
                prevent_initial_call=True)
def update_parallel_coordinates(c_label, dummy):
    global g_dataframe, g_parallel_coord_figure, g_selected_points, g_par_coord_selected_ranges

    g_parallel_coord_figure = generate_parallel_coordinates(
        color_label=c_label)

    if len(g_par_coord_selected_ranges) > 0:
        # Add the selected ranges back
        for axis, values in g_par_coord_selected_ranges.items():
            g_parallel_coord_figure.data[0].dimensions[int(
                axis)].constraintrange = values

    return g_parallel_coord_figure


@g_app.callback(Output('update_parc', 'data'),
                Input('scatter-figure1', 'selectedData'),
                Input('scatter-figure2', 'selectedData'),
                Input('scatter-figure3', 'selectedData'),
                Input('scatter-figure4', 'selectedData'),
                Input('scatter-figure5', 'selectedData'),
                Input('isDataLoaded', 'data'),
                prevent_initial_call=True)
def get_selected_data_parc(scatter_data1, scatter_data2, scatter_data3,
                           scatter_data4, scatter_data5, dummy):
    global g_dataframe, g_selected_points_parc, g_par_coord_selected_ranges

    # Get the selected rows from each scatter plot
    scatter1_points = [p['pointIndex']
                       for p in scatter_data1['points']] if scatter_data1 else []
    scatter2_points = [p['pointIndex']
                       for p in scatter_data2['points']] if scatter_data2 else []
    scatter3_points = [p['pointIndex']
                       for p in scatter_data3['points']] if scatter_data3 else []
    scatter4_points = [p['pointIndex']
                       for p in scatter_data4['points']] if scatter_data4 else []
    scatter5_points = [p['pointIndex']
                       for p in scatter_data5['points']] if scatter_data5 else []

    rows_from_scatters = []
    # Add list of points if they are not empty

    for list_rows in [scatter1_points, scatter2_points, scatter3_points,
                      scatter4_points, scatter5_points]:
        if len(list_rows) > 0:
            rows_from_scatters.append(list_rows)

    # Process points depending if they are one selection or multiple
    if len(rows_from_scatters) > 1:
        g_selected_points_parc = list(
            reduce(set.intersection, [set(x) for x in rows_from_scatters]))
    elif len(rows_from_scatters) == 1:
        g_selected_points_parc = rows_from_scatters[0]
    else:
        g_selected_points_parc = []

    return 0


@g_app.callback(Output('update_scatter', 'data'),
                Input('parcoord-figure', 'restyleData'),
                Input('scatter-figure1', 'selectedData'),
                Input('scatter-figure2', 'selectedData'),
                Input('scatter-figure3', 'selectedData'),
                Input('scatter-figure4', 'selectedData'),
                Input('scatter-figure5', 'selectedData'),
                Input('isDataLoaded', 'data'),
                prevent_initial_call=True)
def get_selected_data_scatters(restyleData, scatter_data1, scatter_data2, scatter_data3,
                               scatter_data4, scatter_data5, dummy):
    global g_dataframe, g_selected_points_sca, g_selected_points_parc, g_par_coord_selected_ranges, g_scatter_selection_list

    # Extract and Update Paarcord ranges
    if restyleData:
        # Format of resytledata: [{'dimension[0].constraintrange': [0,1]}]
        for key, val in restyleData[0].items():
            dim = int(re.split(r'\[|\]', key)[1])
            if val is not None:
                constraintrange = [val[0][0], val[0][1]]
                g_par_coord_selected_ranges.update({str(dim): constraintrange})
            else:
                g_par_coord_selected_ranges.pop(str(dim))

    # Get the selected rows from parallel coordinates
    par_selected_points = []

    if g_par_coord_selected_ranges:
        for key, val in g_par_coord_selected_ranges.items():
            # example {'2': [1.193484642049015, 1.23812416090861]}
            # {'4': [1384.7734780172127, 1864.7713803804386]}
            col = g_dataframe.columns[int(key)]

            # Obtain points from range
            temp_points = g_dataframe[g_dataframe[col].between(
                val[0], val[1])].index

            # Use set logic to find and add points that weren't already present
            tmp_points_set = set(par_selected_points)

            if len(par_selected_points) > 0:
                tmp_points_set.intersection_update(set(temp_points))
            else:
                tmp_points_set.update(set(temp_points))

            par_selected_points = list(tmp_points_set)

    # Get the selected rows from each scatter plot
    scatter1_points = [p['pointIndex']
                       for p in scatter_data1['points']] if scatter_data1 else []
    scatter2_points = [p['pointIndex']
                       for p in scatter_data2['points']] if scatter_data2 else []
    scatter3_points = [p['pointIndex']
                       for p in scatter_data3['points']] if scatter_data3 else []
    scatter4_points = [p['pointIndex']
                       for p in scatter_data4['points']] if scatter_data4 else []
    scatter5_points = [p['pointIndex']
                       for p in scatter_data5['points']] if scatter_data5 else []

    g_scatter_selection_list[0] = scatter_data1['range'] if scatter_data1 and 'range' in scatter_data1.keys() else {
    }
    g_scatter_selection_list[1] = scatter_data2['range'] if scatter_data2 and 'range' in scatter_data2.keys() else {
    }
    g_scatter_selection_list[2] = scatter_data3['range'] if scatter_data3 and 'range' in scatter_data3.keys() else {
    }
    g_scatter_selection_list[3] = scatter_data4['range'] if scatter_data4 and 'range' in scatter_data4.keys() else {
    }
    g_scatter_selection_list[4] = scatter_data5['range'] if scatter_data5 and 'range' in scatter_data5.keys() else {
    }

    rows_from_scatters = []
    # Add list of points if they are not empty

    for list_rows in [scatter1_points, scatter2_points, scatter3_points,
                      scatter4_points, scatter5_points, par_selected_points]:
        if len(list_rows) > 0:
            rows_from_scatters.append(list_rows)

    if len(rows_from_scatters) > 1:
        g_selected_points_sca = list(
            reduce(set.intersection, [set(x) for x in rows_from_scatters]))
    elif len(rows_from_scatters) == 1:
        g_selected_points_sca = rows_from_scatters[0]
    else:
        g_selected_points_sca = []

    return 0


def main_inline(dataframe, debug=False):
    global g_dataframe, g_labels, g_parallel_coord_figure, g_scatter_list, g_customdata

    # Load Data
    g_dataframe = dataframe
    g_labels = [{'label': column, 'value': column}
                for column in g_dataframe.columns]
    g_labels = [{'label': 'None', 'value': 'none'}] + g_labels
    default = g_labels[1]['value']

    g_customdata = np.empty(shape=(len(g_dataframe), len(
        g_dataframe.columns), 1), dtype='object')

    for i in range(len(g_dataframe.columns)):
        g_customdata[:, i] = np.array(g_dataframe.iloc[:, i]).reshape(-1, 1)

    g_parallel_coord_figure = generate_parallel_coordinates(
        color_label=default)

    g_scatter_list = [generate_scatter(g_dataframe.columns[-1],
                                       g_dataframe.columns[-2],
                                       g_dataframe.columns[-3], i)
                      for i in range(5)]

    g_app.layout = get_HTML_Layout(ext_load=False, load_figures=True)

    g_app.run_server(debug=debug)


def main_standalone(debug=True):
    global g_dataframe, g_labels, g_parallel_coord_figure, g_scatter_list

    g_app.layout = get_HTML_Layout(ext_load=True)

    # Loading Data Function
    @g_app.callback(Output('dropdown-parcoord', 'options'),
                    Output('dropdown-parcoord', 'value'),

                    Output('sd1-xaxis', 'options'),
                    Output('sd1-yaxis', 'options'),
                    Output('sd1-color', 'options'),
                    Output('sd1-xaxis', 'value'),
                    Output('sd1-yaxis', 'value'),
                    Output('sd1-color', 'value'),

                    Output('sd2-xaxis', 'options'),
                    Output('sd2-yaxis', 'options'),
                    Output('sd2-color', 'options'),
                    Output('sd2-xaxis', 'value'),
                    Output('sd2-yaxis', 'value'),
                    Output('sd2-color', 'value'),

                    Output('sd3-xaxis', 'options'),
                    Output('sd3-yaxis', 'options'),
                    Output('sd3-color', 'options'),
                    Output('sd3-xaxis', 'value'),
                    Output('sd3-yaxis', 'value'),
                    Output('sd3-color', 'value'),

                    Output('sd4-xaxis', 'options'),
                    Output('sd4-yaxis', 'options'),
                    Output('sd4-color', 'options'),
                    Output('sd4-xaxis', 'value'),
                    Output('sd4-yaxis', 'value'),
                    Output('sd4-color', 'value'),

                    Output('sd5-xaxis', 'options'),
                    Output('sd5-yaxis', 'options'),
                    Output('sd5-color', 'options'),
                    Output('sd5-xaxis', 'value'),
                    Output('sd5-yaxis', 'value'),
                    Output('sd5-color', 'value'),

                    Output('isDataLoaded', 'data'),
                    Input('upload-data', 'contents'),
                    State('upload-data', 'filename'),
                    prevent_initial_call=True)
    def load_data(file_contents, filename):
        global g_dataframe, g_labels, g_parallel_coord_figure, g_scatter_list

        if file_contents is None or file_contents is []:
            pass
        else:
            content_type, content_string = file_contents.split(',')
            decoded = base64.b64decode(content_string)

            try:
                if 'csv' in filename:
                    # Assume that the user uploaded a CSV file
                    g_dataframe = pd.read_csv(
                        io.StringIO(decoded.decode('utf-8')))
                    g_labels = [{'label': column, 'value': column}
                                for column in g_dataframe.columns]
                    g_labels = [{'label': 'None', 'value': 'none'}] + g_labels
            except Exception as e:
                print(e)
                pass
            else:

                default_value = g_labels[-1]['value']
                g_parallel_coord_figure = generate_parallel_coordinates(
                    color_label=default_value)

                g_scatter_list = [generate_scatter(g_dataframe.columns[-1],
                                                   g_dataframe.columns[-2],
                                                   g_dataframe.columns[-3], i)
                                  for i in range(5)]

                return g_labels, default_value, \
                    g_labels[1:], g_labels[1:], g_labels, default_value, default_value, default_value, \
                    g_labels[1:], g_labels[1:], g_labels, default_value, default_value, default_value, \
                    g_labels[1:], g_labels[1:], g_labels, default_value, default_value, default_value, \
                    g_labels[1:], g_labels[1:], g_labels, default_value, default_value, default_value, \
                    g_labels[1:], g_labels[1:], g_labels, default_value, default_value, default_value, \
                    True

    g_app.run_server(debug=debug)


if __name__ == '__main__':
    dataframe = pd.read_csv('opt_out.csv')
    main_inline(dataframe, debug=False)
