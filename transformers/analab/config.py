from reviewlab import Config


class JobConfig(Config):
    def __init__(self,
        job_name,
        model_name,
        out_dir="aruns",
        max_seq_len = 128,
        max_examples=150,
        **kwargs):
        super().__init__(**kwargs)
        self.job_name = job_name
        self.model_name = model_name
        self.out_dir = out_dir
        self.max_seq_len = max_seq_len
        self.max_examples = max_examples


class AttentionJobConfig(JobConfig):
    def __init__(self, layer, **kwargs):
        self.layer = layer
        super().__init__("Attention", **kwargs)

class LatentAspectJobConfig(JobConfig):
    def __init__(self, **kwargs):
        super().__init__("LatentAspect", **kwargs)

class LatentANAJobConfig(JobConfig):
    def __init__(self, **kwargs):
        super().__init__("LatentANA", **kwargs)

class MultiDomainAspectJobConfig(JobConfig):
    def __init__(self, **kwargs):
        super().__init__("MultiDomainAspect", **kwargs)

class DomainAspectJobConfig(JobConfig):
    def __init__(self, **kwargs):
        super().__init__("DomainAspect", **kwargs)

class LatentOTJobConfig(JobConfig):
    def __init__(self, **kwargs):
        super().__init__("LatentOT", **kwargs)

class SentiDiscoveryJobConfig(JobConfig):
    def __init__(self, **kwargs):
        super().__init__("SentiDiscovery", **kwargs)

class AspectDiscoveryJobConfig(JobConfig):
    def __init__(self, **kwargs):
        super().__init__("AspectDiscovery", **kwargs)
