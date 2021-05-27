import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FontSizes:
    ticks: int = 14
    legend: int = 14
    labels: int = 14
    title: int = 15
    subtitle: int = 15
    bar_labels: int = 10


def compute_offsets(no_bars, width):
    if no_bars % 2 == 0:
        # [-3/2 * width, -1/2*width, 1/2 * width, 3/2 * width]
        right_half = [1 / 2 * width + i * width for i in range(int(no_bars / 2))]
        return [-o for o in reversed(right_half)] + right_half
    else:
        right_half = [(i + 1) * width for i in range(math.floor(no_bars / 2))]
        return [-o for o in reversed(right_half)] + [0] + right_half


def select_index(iter, idx, default=None):
    if iter is None:
        return default
    return iter[idx]


def autolabel(rects, plt, threshold, font_sizes, height_offset):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if height < threshold:
            label = f"{height:.2f}\%"
            plt.text(rect.get_x() + rect.get_width() / 2., max(height + height * height_offset),
                     label, rotation=90,
                     ha='center', va='bottom').set_fontsize(font_sizes.bar_labels)


def shared_bar_plots(no_plots, series_per_plot, y_data, nrows=None, ncols=None, width=0.1, labels=None, titles=None,
                     title=None, xticklabels=None, figsize=None, y_label=None, x_label=None, colors=None, alphas=None,
                     default_alpha=0.9, patterns=None, sharey='row', sharex='col', font_sizes=None, y_ax_formatter=None,
                     y_limits=None, legend=True, legend_bbox_to_anchor=(1.05, 1), save_path=None,
                     legend_loc='upper left', legend_ncols=1, subplot_adjustment=None, x_tick_rotation=0,
                     x_tick_bottom=0.2, x_tick_ha='bottom', bar_label=False, bar_label_threshold=10,
                     bar_label_offset=0.1, align_y_labels=True, **kwargs):
    if ncols is None:
        ncols = no_plots
    if nrows is None:
        nrows = math.ceil(no_plots / ncols)
    if titles is not None:
        assert len(titles) == no_plots
    if figsize is None:
        figsize = (10, 7)
    if font_sizes is None:
        font_sizes = FontSizes()

    offsets = compute_offsets(series_per_plot, width)
    fig, axes = plt.subplots(nrows, ncols, sharey=sharey, sharex=sharex, figsize=figsize)
    if title is not None:
        fig.suptitle(title, fontsize=font_sizes.title)

    for plot_idx in range(no_plots):

        curr_row = plot_idx // ncols
        curr_col = (plot_idx - curr_row * ncols) % ncols

        if nrows == 1 or ncols == 1:
            ax = axes[plot_idx]
        else:
            ax = axes[curr_row][curr_col]
        if titles is not None:
            ax.set_title(titles[plot_idx], fontsize=font_sizes.subtitle)
        if y_limits is not None:
            ax.set_ylim(y_limits)

        for series_idx in range(series_per_plot):
            label = select_index(labels, series_idx)
            color = select_index(colors, series_idx)
            alpha = select_index(alphas, series_idx, default=default_alpha)
            hatch = select_index(patterns, series_idx)

            y = y_data(series_idx, plot_idx, **kwargs)
            x = np.arange(len(y)) + offsets[series_idx]

            rects = ax.bar(x, y, align='center', width=width, label=label, color=color, alpha=alpha, hatch=hatch)
            if bar_label:
                autolabel(rects, plt, bar_label_threshold, font_sizes, bar_label_offset)

        ax.tick_params(labelsize=font_sizes.ticks)
        if curr_col == 0 and y_label is not None:
            ax.set_ylabel(y_label, fontsize=font_sizes.labels)
        if curr_row == nrows - 1 and x_label is not None:
            ax.set_xlabel(x_label, fontsize=font_sizes.labels)

        if y_ax_formatter is not None:
            ax.yaxis.set_major_formatter(y_ax_formatter)

    if xticklabels is not None:
        plt.setp(axes, xticks=np.arange(len(xticklabels)), xticklabels=xticklabels)
    plt.tight_layout()

    if subplot_adjustment is not None:
        fig.subplots_adjust(top=subplot_adjustment)

    if legend:
        plt.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc, fontsize=font_sizes.legend, ncol=legend_ncols)

    if x_tick_rotation != 0:
        fig.autofmt_xdate(bottom=x_tick_bottom, rotation=x_tick_rotation, ha=x_tick_ha)

    if align_y_labels:
        fig.align_ylabels(axes.reshape(-1))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
