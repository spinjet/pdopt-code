# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:20:45 2022

A visualisation tool to explore the effects of the requirements
over the general design space.

@author: s345001
"""

__author__ = 'Andrea Spinelli'
__copyright__ = 'Copyright 2021, all rights reserved'
__status__ = 'Development'


# Standard Library Imports
import base64
import pickle as pk
import io
import re
from functools import reduce


# Third-party imports
import numpy as np
import pandas as pd

import dash
from dash import dcc, html
#import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
import numpy as np

import plotly.io as pio

from .data import DesignSpace, ExtendableModel, ContinousParameter
from .exploration import  ProbabilisticExploration, generate_input_samples

import matplotlib.pyplot as plt
import numpy as np

from HE_Model import model

pio.renderers.default='browser'

# Scatter Sizes
SCATTER_SIZES = ['45vw', '45vw', '31vw', '31vw', '31vw']

# Color Schemes
SCATTER_COLOR  = 'rainbow'
PARCOORD_COLOR  = 'turbo'
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
                'rgb(122,4,2)',]

STYLESHEETS = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

NONE_LABELS = [{'label' : 'None', 'value' : 'none'}]

##### Do the data generation using the loaded exploration file #####
exploration = pk.load(open('exp_test.pk','rb'))

app_layout = html.Div([
    
    # Title
    html.H1(children='P-DOPT Design Space Exploration'),
    html.H5(children='This webapp allows to visualize .csv outputs from P-DOPT'),
    
    html.Div(id='debug'),
             
    html.Br(),
    
    # Parallel Coordinates
    html.Div(id='parcoord-div',
             children=[
                 
                html.Div(children=[
                    
                dcc.Graph(id='parcoord-figure',
                          figure= go.Figure(),
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
             ),]),
    
    html.Br(),
               
    html.Div(children=[
            
            # Scatter Plot 1
            html.Div(id='sd1',children=[
                
                html.Div([
                    dcc.Graph(id='scatter-figure1',
                              figure=go.Figure(),
                              style={'height': '47vw','width':  '47vw'}
                              )
                    ]),
            
            # Drop Down menus to selects axes and colors
            html.Div(children=[
                html.Div(['X axis',
                      
                      dcc.Dropdown(
                          id='sd1-xaxis',
                          options=NONE_LABELS,
                          value=NONE_LABELS[0]['value'],
                          ),
                        ],
                     style={'width': '30%', 
                            'display': 'inline-block', 
                            'padding': '5px'}
                     ),

                html.Div(['Y axis',
                      
                      dcc.Dropdown(
                          id='sd1-yaxis',
                          options=NONE_LABELS,
                          value=NONE_LABELS[0]['value'],
                          ),
                      ], 
                     style={'width': '30%', 
                            'display': 'inline-block', 
                            'padding': '5px'}),
            
                html.Div(['Color axis',
                
                      dcc.Dropdown(
                          id='sd1-color',
                          options=NONE_LABELS,
                          value=NONE_LABELS[0]['value'],
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
                       'float':'left',
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
            ]),
    
        html.Div(id='options',children=[
            ],),
            
    html.Br(),
    html.Div(id='copyright', children=['V. 0.1, Copyright Andrea Spinelli']),
])

g_app = dash.Dash(__name__, external_stylesheets=STYLESHEETS)
g_app.layout = app_layout

if __name__ == '__main__':
    print('ciao')
    # g_app.run_server(debug=False)