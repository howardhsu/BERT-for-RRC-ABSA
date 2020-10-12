import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
import matplotlib
import random

from torch.optim.optimizer import Optimizer, required
from torch.distributions import Categorical

from matplotlib import pyplot as plt

from .plot import Plot


class NeuronPlot(Plot):
    def __init__(self, fn, width=25, height=15):
        super().__init__(fn, width, height)
        self.width = width
        self.height = height

    def plot(self, log_reg):
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.plot(range(len(log_reg.coef_[0])), log_reg.coef_.T)
        plt.xlabel('Neuron Index', size=20)
        plt.ylabel('Neuron Weight', size=20)
