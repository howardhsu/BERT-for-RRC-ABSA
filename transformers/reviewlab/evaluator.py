
import json
import os
import sklearn.metrics
import numpy as np
import glob
import time
import json
import math
import random
import argparse
import logging

import xml.etree.ElementTree as ET

from collections import Counter, namedtuple
from subprocess import check_output
from reviewlab.seq_utils import compute_metrics_absa
from . import metric


logger = logging.getLogger(__name__)


class Evaluator(object):
    def batch_eval(self, config_dir):
        configs = glob.glob(os.path.join(config_dir, "*.json") )
        scores = {}
        for cfg in configs:
            with open(cfg) as f:
                config = json.load(f)
            scores[cfg.split("/")[-1]] = self.single_run(cfg, config)
        with open(config_dir + "result.json", "w") as fw:
            json.dump(scores, fw, indent = 4)
    
    def single_eval(self, cfg):
        with open(cfg) as f:
            config = json.load(f)
            scores = self.single_run(cfg, config)
        with open(os.path.join(config["output_dir"], "result.json"), "w") as fw:
            json.dump(scores, fw, indent = 4)

    def succ(self, run_config):
        for test_file in run_config.test_file:
            if not os.path.exists( os.path.join(run_config.output_dir, "predictions.json")):
                return False
        return True
    
    def single_run(self, name, config):
        """evaluate on a single .json config file.
        """

        summary = {}
        for test_file in config["test_file"]:
            scores = []
            for seed in range(1, config["run"]+1):
                run_config = dict(config)
                run_config["seed"] = seed
                run_config["output_dir"] = os.path.join(run_config["output_dir"], str(seed))
                run_config = namedtuple("run_config", run_config.keys())(*run_config.values())

                if not self.succ(run_config):
                    print("incomplete running of", name)
                    return

                fn = os.path.join(run_config.output_dir, "predictions.json")
                with open(fn) as f:
                    preds = json.load(f)
                    
                eval_class = getattr(metric, run_config.task.upper() + "Metric")
                
                scores.append(eval_class.evaluate(run_config, preds) )

            scores = np.array(scores).mean(axis=0)
            mtr = {eval_class.metric_name(ix): score for ix, score in enumerate(scores)}
            summary[test_file] = mtr
        return summary
