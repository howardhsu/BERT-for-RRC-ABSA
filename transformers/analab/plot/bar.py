
import numpy as np
import matplotlib.ticker as ticker

from .plot import Plot


class BarPlot(Plot):
    def __init__(self, ax):
        self.ax = ax

    def __call__(self, data, x_labels, y_labels):
        _COLOR = ["b", "g", "r"]
        
        y_pos = np.arange(len(x_labels))
        for bar_idx, (data_label, act_cnt) in enumerate(data):
            pos = [_y_pos -0.2 + bar_idx * 0.2 for _y_pos in y_pos]
            self.ax.bar(pos, act_cnt, width=0.2, color=_COLOR[bar_idx], align='center', alpha=0.7)
        self.ax.tick_params(labelsize=15)
        self.ax.xticks(y_pos, x_labels, rotation=45)
        self.ax.ylabel(y_labels, fontsize=15)
        self.ax.title(tt, fontsize=20)
        
        self.ax.legend([], prop={'size': 30})
        