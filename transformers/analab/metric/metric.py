
import numpy as np
import sklearn.metrics


class Metric(object):
    """An abstract class for (online) evaluation metric. 
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.outputs = []
        self.targets = []

    def collect(self, output, target):
        self.outputs.append(output)
        self.targets.append(target)

    def aggregate(self, **kwargs):
        raise NotImplementedError

