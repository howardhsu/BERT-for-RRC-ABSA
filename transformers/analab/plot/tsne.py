import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
import matplotlib
import random

from collections import defaultdict
from sklearn.manifold import TSNE
from torch.optim.optimizer import Optimizer, required
from torch.distributions import Categorical

from matplotlib import pyplot as plt

from .plot import Plot


class TSNEPlot(Plot):
    def __init__(self, fn, width=10, height=8):
        super().__init__(fn, width, height)
        plt.axis("off")
        self.width = width
        self.height = height


    def plot(self, embs, labels):
        tsne = TSNE(n_components=2, random_state=2)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(embs)

        x_coords = Y[:, 0]
        y_coords = Y[:, 1]

        for ex_idx in range(len(x_coords)):
            plt.scatter(x_coords[ex_idx], y_coords[ex_idx])

        for label, x, y in zip(labels, x_coords, y_coords):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

        plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
        plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)


class TypedTSNEPlot(Plot):
    def __init__(self, fn, width=8, height=6):
        super().__init__(fn, width, height)
        plt.axis("off")
        self.width = width
        self.height = height


    def plot(self, embs, phrases, labels, types, color_map, annotate=True):
        tsne = TSNE(n_components=2, random_state=2)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(embs)

        x_coords = Y[:, 0]
        y_coords = Y[:, 1]

        typed_ex_idxs = defaultdict(list)
        for ex_idx, label_type in enumerate(labels):
            typed_ex_idxs[label_type].append(ex_idx)

        for label_type in types:
            ex_idxs = typed_ex_idxs[label_type]
            plt.scatter(x_coords[ex_idxs], y_coords[ex_idxs], color=color_map[label_type], label=label_type)

        if annotate:
            for ex_idx, (phrase, x, y) in enumerate(zip(phrases, x_coords, y_coords)):
                plt.annotate(phrase, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=12)

        plt.xlim(x_coords.min()-0.00005, x_coords.max()+0.00005)
        plt.ylim(y_coords.min()-0.00005, y_coords.max()+0.00005)
        plt.legend()
