import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def box_violin_plot(scores, cmap='tab10', ylabel=None, xlabel=None, title=None, fontsize=14):
    """ Plot box-violin plot to compare list of values within scores """

    # plot violin plot
    vplot = plt.violinplot(scores, showextrema=False);

    # set colors for biolin plot
    num_colors = len(scores)
    cmap = cm.ScalarMappable(cmap=cmap)
    color_mean = np.linspace(0, 1, num_colors)      
    for patch, color in zip(vplot['bodies'], cmap.to_rgba(color_mean)):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        
  # plot box plot
    bplot = plt.boxplot(scores, notch=True, patch_artist=True, widths=0.2,    
                      medianprops=dict(color="black",linewidth=2), showfliers=False,
                      showmeans=True, meanprops=dict(markerfacecolor="red", markeredgecolor="red", linewidth=2))

  # set colors for box plot
    for patch, color in zip(bplot['boxes'], cmap.to_rgba(color_mean)):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')

  # set up plot params
    ax = plt.gca();
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize);
    if xlabel is not None:
        plt.xticks(range(1,num_colors+1), xlabel, fontsize=fontsize, rotation=45, horizontalalignment="right");
    if title is not None:
        plt.title(title, fontsize=fontsize)

    return ax




def plot_glifac(ax, correlation_matrix, filter_labels, vmin=-0.5, vmax=0.5):
    ax.set_xticks(list(range(len(filter_labels))))
    ax.set_yticks(list(range(len(filter_labels))))
    ax.set_xticklabels(filter_labels, rotation=90)
    ax.set_yticklabels(filter_labels)
    c = ax.imshow(correlation_matrix, cmap='bwr_r', vmin=vmin, vmax=vmax)
    return ax, c