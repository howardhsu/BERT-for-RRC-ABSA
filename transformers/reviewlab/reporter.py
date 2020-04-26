import json
import os
import glob
import sklearn.metrics
import numpy as np
import time
import math
import random
import csv
import logging

from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class Reporter(object):
    def __init__(self, run_dir="ft_runs"):
        self.run_dir = run_dir

    def run(self):
        data = self._load_results()
        tasks, baselines = self._print(data)
        self._to_csv(tasks, baselines)

    def _load_results(self):
        data = {}
        for taskdir in os.listdir(self.run_dir):
            result_file = os.path.join(self.run_dir, taskdir, "result.json")
            if os.path.exists(result_file):
                with open(result_file) as f:
                    result = json.load(f)
                data[taskdir] = result
        return data

    def _print(self, data):
        tasks = defaultdict(dict)

        print("------------------------------------------------------")
        baselines = set()
        for rec in data:
            segs = rec.split("_")
            task = "_".join(segs[:3])
            baseline = "_".join(segs[3:])
            baselines.add(baseline)
            tasks[task][baseline] = data[rec]

        for task in tasks:
            print(task)
            for baseline in baselines:
                metrics = tasks[task][baseline]["test.json"]
                print(baseline, " "* (30 - len(baseline)), "\t".join(["{} = {} ".format(metric, "%.4f" % round(metrics[metric], 4)) for metric in metrics]))
            print("\n")
        print("-----------------------------------------------------")
        return tasks, baselines


    def _to_csv(self, tasks, baselines, max_col = 10):
        now = datetime.now()

        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

        with open(os.path.join("ft_runs", dt_string + "_report.csv"), "w") as fw:
            # fieldnames = ['task', 'baseline'] + ['metric'+ix for ix in range(10) ]
            # writer = csv.DictWriter(fw, fieldnames=fieldnames)
            writer = csv.writer(fw)
                
            for task in tasks:
                for ix, baseline in enumerate(baselines):
                    metrics = tasks[task][baseline]["test.json"]
                    if ix == 0:
                        cols = [task] + list(metrics.keys())
                        padded_cols = cols + ["_"] * (max_col - len(cols))
                        writer.writerow(padded_cols)
                        
                    cols = [baseline] + [round(metrics[metric], 4) for metric in metrics]
                    padded_cols = cols + ["_"] * (max_col - len(cols))
                    writer.writerow(padded_cols)