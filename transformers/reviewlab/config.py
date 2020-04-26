import os
import json
import copy

class Config(object):
    """
    Base class for all configurations.
    """
    def __init__(self, **kwargs):
        self.config_cls = self.__class__.__name__

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict):
        config = cls(**config_dict)
        return config

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_file(self, json_file):
        with open(json_file, "w") as fw:
            json.dump(self.to_dict, fw)


class TaskConfig(Config):
    def __init__(self, task, year, domain, **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.year = year
        self.domain = domain
        self.data_dir = os.path.join("data/ft", task, year, domain)
        

class TrainConfig(Config):
    def __init__(self, 
        max_seq_length=128,
        train_batch_size=32,
        learning_rate=3e-5,
        run=10,
        eval_batch_size=32,
        remove_model=True,
        num_train_epochs=8,
        fp16=True,
        do_lower_case=True,
        adam_epsilon=1e-8,
        fp16_opt_level="O1",
        max_grad_norm=1.0,
        weight_decay=0.0,
        warmup_steps=0,
        no_cuda=False,
        n_gpu=1,
        device=0,
        **kwargs
    ):
        super().__init__(**kwargs)    
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size 
        self.learning_rate = learning_rate 
        self.run = run
        self.eval_batch_size = eval_batch_size 
        self.remove_model = remove_model 
        self.num_train_epochs = num_train_epochs 
        self.fp16 = fp16
        self.do_lower_case = do_lower_case 
        self.adam_epsilon = adam_epsilon 
        self.fp16_opt_level = fp16_opt_level 
        self.max_grad_norm = max_grad_norm 
        self.weight_decay = weight_decay 
        self.warmup_steps = warmup_steps 
        self.no_cuda = no_cuda 
        self.n_gpu = n_gpu
        self.device = device


class BaselineConfig(Config):
    def __init__(self, 
        baseline, 
        model_type="bert", 
        model_name_or_path="bert-base-uncased", 
        test_file = ["test.json"],
        **kwargs):
        super().__init__(**kwargs)
        self.baseline = baseline
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.test_file = test_file
        