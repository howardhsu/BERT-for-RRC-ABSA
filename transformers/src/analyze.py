import argparse

import json
import os

from analab import AttentionJob, \
                   LatentAspectJob, \
                   MultiDomainAspectJob, \
                   LatentANAJob, \
                   DomainAspectJob, \
                   LatentOTJob, \
                   SentiDiscoveryJob, \
                   AspectDiscoveryJob

from analab import AttentionJobConfig, \
                   LatentAspectJobConfig, \
                   MultiDomainAspectJobConfig, \
                   LatentANAJobConfig, \
                   DomainAspectJobConfig, \
                   LatentOTJobConfig, \
                   SentiDiscoveryJobConfig, \
                   AspectDiscoveryJobConfig

from reviewlab import TaskConfig

a = {
    "Attention": {
        "job": AttentionJob,
        "config": TaskConfig("e2e", "14", "laptop")
    },
    "LatentANA": {
        "job": LatentANAJob,
        "config": TaskConfig("e2e", "14", "laptop")
    },
    "MultiDomainAspect": {
        "job": MultiDomainAspectJob,
        "config": [TaskConfig("e2e", "14", "laptop"), TaskConfig("e2e", "union", "rest")]
    },
    "DomainAspect": {
        "job": DomainAspectJob,
        "config": [TaskConfig("e2e", "14", "laptop"), TaskConfig("e2e", "union", "rest")]
    },
    "LatentOT": {
        "job": LatentOTJob,
        "config": TaskConfig("e2e", "14", "laptop"),
    },
    "AspectDiscovery": {
        "job": AspectDiscoveryJob,
        "config": [TaskConfig("e2e", "14", "laptop"), TaskConfig("e2e", "union", "rest")]
    },
}

JOB_CLS = {
    "SentiDiscovery": {
        "job": SentiDiscoveryJob,
        "config": TaskConfig("e2e", "14", "laptop"),
    },
}


MODELS = [
    # "bert-base-uncased",
    # "activebus/BERT-PT_DOMAIN",
    # "activebus/BERT_Review",
    "activebus/BERT-XD_Review",
]

CONFIG_CLS = {
    "Attention": (AttentionJobConfig, {"layer": list(range(12))}),
    "LatentAspect": (LatentAspectJobConfig, {}),
    "LatentANA": (LatentANAJobConfig, {}),
    "MultiDomainAspect": (MultiDomainAspectJobConfig, {}),
    "DomainAspect": (DomainAspectJobConfig, {}),
    "LatentOT": (LatentOTJobConfig, {}),
    "SentiDiscovery": (SentiDiscoveryJobConfig, {}),
    "AspectDiscovery": (AspectDiscoveryJobConfig, {}),
}


def main():
    for job_name, JOB in JOB_CLS.items():
        for model_name in MODELS:
            data_config = JOB["config"]
            if not isinstance(data_config, list):
                model_uri = model_name.replace("DOMAIN", data_config.domain)
            else:
                model_uri = model_name
            config_cls, param = CONFIG_CLS[job_name]
            job_config = config_cls(**param, model_name=model_uri)
            job = JOB["job"](job_config, data_config)
            job.run()


if __name__ == '__main__':
    main()
