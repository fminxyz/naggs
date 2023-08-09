# Source https://wandb.ai/matt24/vit-snacks-sweeps/reports/Hyperparameter-Search-for-HuggingFace-Transformer-Models--VmlldzoyMTUxNTg0

#======== IMPORTS ========
from datasets import load_dataset
from transformers import AutoFeatureExtractor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, \
    ToTensor, Resize
from transformers import DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, \
    Trainer
import numpy as np
from datasets import load_metric
from transformers import Trainer, AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD
import torch as T
from typing import List, Optional
from pathlib import Path
import wandb
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#======== PARAMETERS ========
methods = ["AdamW", "NagGS"]
DATA_FRACTION = 1
n_epochs = 2
batch_size = 16

sweep_config = {
    "method": "bayes",
    "metric":{
        "name": "accuracy",
        "goal": "maximize"
    }   
}

parameters_dict = {}
parameters_dict["NagGS"] = {
    'Data fraction': {
        "value": DATA_FRACTION, # 1 stands for the whole dataset,
        },
    'n_epochs': {
        "value": n_epochs,
        },
    'batch_size': {
        "value": batch_size,
        },
    "learning_rate": {
        'distribution': 'log_uniform_values',
        'min': 1e-4,
        'max': 1e-1
    },
    "mu": {
        "min": 1e-2,
        "max": 1e1
    },
    "gamma":{
        "min": 1e-2,
        "max": 1e1
    }
}

parameters_dict["SGD with momentum"] = {
    'Data fraction': {
        "value": DATA_FRACTION, # 1 stands for the whole dataset,
        },
    'n_epochs': {
        "value": n_epochs,
        },
    'batch_size': {
        "value": batch_size,
        },
    "learning_rate": {
        'distribution': 'log_uniform_values',
        'min': 1e-4,
        'max': 1e-1
    },
    "momentum": {
        "distribution": "uniform",
        "min": 0.8,
        "max": 1.0
    }
}

parameters_dict["AdamW"] = {
    'Data fraction': {
        "value": DATA_FRACTION, # 1 stands for the whole dataset,
        },
    'n_epochs': {
        "value": n_epochs,
        },
    'batch_size': {
        "value": batch_size,
        },
    "learning_rate": {
        'distribution': 'log_uniform_values',
        'min': 1e-6,
        'max': 1e-2
    },
    "beta1": {
        "distribution": "uniform",
        "min": 0.8,
        "max": 1.0
    },
   "beta2": {
        "distribution": "uniform",
        "min": 0.99,
        "max": 1.0
    },
}

WANDB_ENTITY = "YOUR_ENTITY"
WANDB_PROJECT = "ViT experiments"
WANDB_GROUP = f"Hyperparameter search"
#======== DATASET LOADING ========
num_train_ims = int(DATA_FRACTION*101000)
percent_data = int(DATA_FRACTION*100)
dataset = load_dataset(
    "food101", 
    cache_dir="/raid/cache", 
    split=f"train[:{percent_data}%]+validation[:{percent_data}%]"
    )
dataset = dataset.train_test_split(test_size=0.2)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

feature_extractor = \
    AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

normalize = Normalize(mean=feature_extractor.image_mean, 
                    std=feature_extractor.image_std)

_transforms = Compose([Resize(tuple(feature_extractor.size.values())), 
                        ToTensor(), 
                        normalize])

def transforms(examples):
    examples["pixel_values"] = \
        [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

dataset = dataset.with_transform(transforms)

data_collator = DefaultDataCollator()

#======== MODEL LOADING ========
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_init():
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    return model

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), 
        references=p.label_ids
        )

#======== OPTIMIZER LOADING ========
class NagGS(T.optim.Optimizer):
    """Class NagGS implements algorithm with update on gamma which stands for
    the semi-implicit integration of the Nesterov Accelerated Gradient (NAG)
    flow.
    Arguments
    ---------
        params (collection): Collection of parameters to optimize.
        lr (float, optional): Learning rate (or alpha).
        mu (float, optional): Momentum mu.
        gamm (float, optional): Gamma factor.
    """

    def __init__(self, params, lr=1e-2, mu=1.0, gamma=1.0):
        if lr < 0.0:
            raise ValueError(f"alpha: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid mu: {mu}")
        if gamma < 0.0:
            raise ValueError('Parameter gamma should be non-nevative.')

        defaults = dict(lr=lr, mu=mu, gamma=gamma)
        super().__init__(params, defaults)

    @T.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments
        ---------
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with T.enable_grad():
                loss = closure()

        for group in self.param_groups:
            state = []
            params = []
            grads = []
            gammas = []

            for p in filter(lambda x: x.grad is not None, group['params']):
                if len(param_state := self.state[p]) == 0:
                    param_state['state'] = T.clone(p).detach()
                state.append(param_state['state'])
                params.append(p)
                grads.append(p.grad)
                gammas.append(param_state.get('gamma', group['gamma']))

            nag_gs(state, params, grads, group['lr'], gammas, group['mu'])

            for param, gamma in zip(params, gammas):
                self.state[param]['gamma'] = gamma

        return loss

def nag_gs(state: List[T.Tensor], params: List[T.Tensor],
           grads: List[T.Tensor], alpha: float, gammas: List[float],
           mu: float):
    """Function nag-gs performs full NAG-GS algorithm computation.
    """
    gammas_out = []
    for gamma, gs, xs, vs in zip(gammas, grads, params, state):
        # Update constants
        a = alpha / (alpha + 1)
        b = alpha * mu / (alpha * mu + gamma)
        gamma = (1-a)*gamma + a*mu
        gammas_out.append(gamma)
        
        # Update state v
        vs.mul_(1 - b)
        vs.add_(xs, alpha=b)
        vs.add_(gs, alpha=-b / mu)
        
        # Update parameters x
        xs.mul_(1 - a)
        xs.add_(vs, alpha=a)

    return gammas_out

#======== TRAINING ========
for method in methods:
    sweep_config['parameters'] = parameters_dict[method]
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)

    def train(config=None):
        with wandb.init(config=config):
            # set sweep configuration
            config = wandb.config

            # set optimizers
            model = model_init()
            if method == "NagGS":
                opt = NagGS(
                    model.parameters(),
                    lr=config.learning_rate,
                    mu=config.mu,
                    gamma=config.gamma
                    )
            elif method == "SGD with momentum":
                opt = SGD(
                    model.parameters(),
                    lr=config.learning_rate,
                    momentum=config.momentum
                    )
            elif method == "AdamW":
                opt = AdamW(
                    model.parameters(),
                    lr=config.learning_rate,
                    betas=(config.beta1, config.beta2)
                    )

            optimizers = opt, get_linear_schedule_with_warmup(
                opt, 
                num_warmup_steps=500, 
                num_training_steps=10000
                )

            # set training arguments
            training_args = TrainingArguments(
                output_dir="./output",
                report_to='wandb',  # Turn on Weights & Biases logging
                num_train_epochs=config.n_epochs,
                per_device_train_batch_size=config.batch_size,
                evaluation_strategy='steps',
                save_steps=100,
                eval_steps=100,
                logging_steps=10,
                save_total_limit=2,
                logging_strategy='epoch',
                load_best_model_at_end=True,
                remove_unused_columns=False,
                fp16=True
            )

            # define training loop
            trainer = Trainer(
                model = model,
                # model_init=model_init,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                compute_metrics=compute_metrics,
                optimizers=optimizers,
                tokenizer=feature_extractor,
            )

            # start training loop
            trainer.train()

    wandb.agent(sweep_id, train, count=300)   