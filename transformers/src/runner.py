
import os
import argparse
import json
import glob
import random
import numpy as np
import torch
from collections import namedtuple
import reviewlab
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)


class RunManager(object):

    def succ(self, run_config):
        for test_file in run_config.test_file:
            if not os.path.exists( os.path.join(run_config.output_dir, "predictions.json")):
                return False
        return True
            
    def run_seed(self, config, seed):
        run_config = dict(config)
        run_config["seed"] = seed
        run_config["output_dir"] = os.path.join(run_config["output_dir"], str(seed))

        run_config = namedtuple("run_config", run_config.keys())(*run_config.values())

        os.makedirs(run_config.output_dir, exist_ok = True)
        if self.succ(run_config):
            return

        with open(os.path.join(run_config.output_dir, "config.json"), "w") as fw:
            json.dump(config, fw)

        reviewlab.util.set_seed(run_config)

        trainer = getattr(reviewlab, run_config.task.upper() + "Trainer")()
        trainer.train(run_config)

        for test_file in run_config.test_file:
            trainer.test(run_config, test_file)
        if run_config.remove_model:
            os.remove(os.path.join(run_config.output_dir, "model.pt"))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required = True, type = str)
    parser.add_argument("--seed", required = True, type = int)
    args = parser.parse_args()
    
    mgr = RunManager()
    with open(args.config) as f:
        config = json.load(f)
    mgr.run_seed(config, args.seed)
    