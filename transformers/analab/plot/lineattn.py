import pytest

import argparse
import gym
import numpy as np
from itertools import count

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


class LineAttn(object):
    def __init__(self, plt, width=3, height=6, color="#9b59b6"):
        plt.figure(figsize=(width, height))
        plt.axis("off")
        self.plt = plt
        self.width = width
        self.height = height
        self.color = color
        
    def __call__(self, words):
        word_height = 1. / len(words)
        
        for position, word in enumerate(words):
            plt.text(.1, 1. - float(position) * word_height, word, ha="right", va="center")
            plt.text(.9, 1. - float(position) * word_height, word, ha="left", va="center")
        
        for left_idx in range(len(words)):
            for right_idx in range(len(words)):
                x_points = [.1, .9]
                y_points = [1. - float(left_idx) * word_height, 1. - float(right_idx) * word_height]
                plt.plot(x_points, y_points, color=self.color, linewidth=1, alpha=1.)
        plt.savefig("line_attn.png")


class TestPlot(object):

    def test_plot_text(self):
        
        words = ["[CLS]", "I", "like", "ice", "cream", "[SEP]"]
        plot = LineAttn(plt)
        
        plot(words)
        assert False


if __name__ == '__main__':
    pytest.main([__file__])
    