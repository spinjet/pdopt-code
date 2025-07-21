# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:35:49 2025

@author: s345001
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_parc(df_in, query_list=None, sel_idxs=None, sel_labels=None,
              caxis=None, axis_names=None, figure_num=None,
              colormap=mpl.cm.viridis, alpha=0.7,
              invert_order=False, line_thickness=2,
              sel_cm='Set1',
              color_list=None,
              label_alignment='center',
              label_rotation=0,
              label_size=15,
              tick_size=15,
              fig_size=(18,6),
              ax_ranges=None):
    
    df = df_in.copy()
    ## Set a selection based on query
    df['sel'] = 0
    s_count = 0
    if query_list is not None and len(query_list) > 0:
        for i, query in enumerate(query_list):
            sel_idx = df.query(query).index
            s_count += len(sel_idx)
            df.loc[sel_idx, 'sel'] = i + 1

        df = df.sort_values(by='sel')
    elif sel_idxs is not None and len(sel_idxs) > 0:
        for i, sel_idx in enumerate(sel_idxs):
            s_count += len(sel_idx)
            df.loc[sel_idx, 'sel'] = i + 1
            
        df = df.sort_values(by='sel')

    # Data Preparation
    if axis_names is not None:
        axis_names = axis_names.copy()
        ynames = axis_names if axis_names is not None else df.columns.tolist()[0:len(df.columns)-1]
    else:
        ynames = df.columns.tolist()[0:len(df.columns)-1]
        
    ys = df.to_numpy()[:,0:len(df.columns)-1]
    z_sel = df['sel'].to_numpy()
    
    sel_ids = df['sel'].unique()
    
    # Define ranges of the axes
    if ax_ranges is None:
        ymins = ys.min(axis=0)
        ymaxs = ys.max(axis=0)
    else:
        ymins = ax_ranges[0]
        ymaxs = ax_ranges[1]
    dys = ymaxs - ymins
    #ymins -= dys * 0.05  # add 5% padding below and above
    #ymaxs += dys * 0.05
    
    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]
    
    #if the color axis is selected, mark the label
    if caxis is not None:
        if caxis == -1:
            caxis = len(df_in.columns)-1
        ynames[caxis] = f'*{ynames[caxis]}*'
        

    if figure_num is not None:
        fig, host = plt.subplots(num=figure_num, figsize=fig_size)
    else:
        fig, host = plt.subplots(figsize=fig_size)
    
    z_ticks = np.linspace(ymins[0], ymaxs[0], 5)
    
    # Construct the vertical axes
    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[0], ymaxs[0])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=tick_size)
        
        if len(str(int(ymaxs[i]))) > 2:
            y_labels = [f'{x:.0f}' for x in np.linspace(ymins[i], ymaxs[i], len(z_ticks))]
        else:
            y_labels = [f'{x:.3f}' for x in np.linspace(ymins[i], ymaxs[i], len(z_ticks))]
        ax.set_yticks(z_ticks)
        ax.set_yticklabels(y_labels)
        
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
    
    # remove the horizontal one, apply axis labels
    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, horizontalalignment=label_alignment,
                         fontsize=label_size, rotation=label_rotation)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()


    ## Manage colors
    colors = [(0.5, 0.5, 0.5, 0.3) for j in range(ys.shape[0])]
    
    if caxis is not None:
        # If there is a selected color axis
        cmap = colormap 
        norm = mpl.colors.Normalize(vmin=zs[:,caxis].min(), vmax=zs[:,caxis].max())
        
        if z_sel.any():
            # There is a selection
            for j, val in enumerate(z_sel):
                if val > 0:
                    colors[j] = cmap(norm(zs[j, caxis]))
        else:
            colors = [cmap(norm(zs[j, caxis])) for j in range(len(zs))]
        
        legend_handles = None
        
    else:
        if color_list is not None:
            cmap = ListedColormap(color_list)
            for j, val in enumerate(z_sel):
                if val > 0:
                    colors[j] = cmap(val - 1)
        
        else:
            cmap = mpl.cm.get_cmap(sel_cm, len(sel_ids))
            for j, val in enumerate(z_sel):
                if val > 0:
                    colors[j] = cmap(val - 1)
        
        if len(sel_ids) > 1:
            if 0 in sel_ids:
                if sel_labels is not None:
                    legend_handles = [Patch(color=cmap(j-1), label=sel_labels[j-1]) for j in sel_ids[1:]]
                else:
                    legend_handles = [Patch(color=cmap(j-1), label=f'selection {j}') for j in sel_ids[1:]]
            else:
                if sel_labels is not None:
                    legend_handles = [Patch(color=cmap(j-1), label=sel_labels[j-1]) for j in sel_ids[0:]]
                else:
                    legend_handles = [Patch(color=cmap(j-1), label=f'selection {j}') for j in sel_ids[0:]]                
        else:
            legend_handles = None
            
    ## Plot the lines
    
    if invert_order:
        for j in range(ys.shape[0]-1,0,-1):
            host.plot(range(ys.shape[1]), zs[j,:], 
                      color=colors[j], alpha=alpha,
                      lw=line_thickness)
    else:
        for j in range(ys.shape[0]):
            host.plot(range(ys.shape[1]), zs[j,:], 
                      color=colors[j], alpha=alpha,
                      lw=line_thickness)        
        
    ## Add selected points counter
    s_count = s_count if s_count > 0 else len(df)
    annotation = f'{s_count}/{len(df)} datapoints'
    
    if legend_handles is not None:
        legend_handles = [Patch(color='none', label=annotation)] + legend_handles 
    else:
        legend_handles = [Patch(color='none', label=annotation)]
    
    # plt.text(annotation, (1,-0.1), 
    #              xycoords='axes fraction', fontsize=13)
    #plt.text(0,-0.05, annotation, transform=axes[0].transAxes, fontsize=13)
    
    if legend_handles is not None:
        host.legend(handles=legend_handles, ncol=len(sel_ids)+1,
                    loc='upper left', frameon=False, fontsize=13,
                    bbox_to_anchor=(-0.1,-0.015))
    plt.tight_layout(pad=1.1)
    
    return fig, axes


exp_original = pd.read_csv("test_case_linear/exp_results.csv")
exp_bn = pd.read_csv("test_case_linear2/exp_results.csv")

opt_original = pd.read_csv("test_case_linear/opt_results.csv")
opt_bn = pd.read_csv("test_case_linear2/opt_results2.csv")


tmp1 = exp_original[['climb_h0','climb_h1','cruise_h0','cruise_h1','P']]
tmp2 = exp_bn[['climb_h0','climb_h1','cruise_h0','cruise_h1','P']]


plt.figure()
s = plt.scatter(opt_original['Mf'], opt_original['M_NOx'], c=opt_original['set_id'], marker='s')
plt.scatter(opt_bn['Mf'], opt_bn['M_NOx'], c=opt_bn['set_id'], marker='s')
plt.colorbar(s)