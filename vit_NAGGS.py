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
from transformers import Trainer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch as T
from typing import List, Optional
from pathlib import Path
import wandb
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#======== PARAMETERS ========
config={
    "lr": 0.07929,
    "gamma": 0.3554,
    "mu": 0.1301,
    "Data fraction": 1, # 1 stands for the whole dataset,
    "n_epochs": 25,
    "batch_size": 16,
}

WANDB_ENTITY = "YOUR_ENTITY "
WANDB_PROJECT = "ViT full scale experiments"
WANDB_GROUP = f"Alive learning rate"
WANDB_RUN_NAME = f"NagGS. Best HP on {int(config['Data fraction']*100)}% of data"
#======== DATASET LOADING ========
num_train_ims = int(config["Data fraction"]*101000)
percent_data = int(config['Data fraction']*100)
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
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), 
        references=p.label_ids
        )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
config["# of trainable parameters"] = count_parameters(model)
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

opt = NagGS(
        model.parameters(), 
        lr=config["lr"], 
        gamma=config["gamma"],
        mu=config["mu"]
        )

optimizers = opt, get_linear_schedule_with_warmup(
    opt, 
    num_warmup_steps=500, 
    num_training_steps=int(0.8*101000/config["batch_size"]*config["n_epochs"])
    )

#======== TRAINING ========
wandb.init(project=WANDB_PROJECT, 
           name=WANDB_RUN_NAME,
           group=WANDB_GROUP,
           entity=WANDB_ENTITY,
           config=config)

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=config["batch_size"],
    evaluation_strategy="steps",
    num_train_epochs=config["n_epochs"],
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to='wandb',
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
    optimizers=optimizers,
    compute_metrics=compute_metrics
)

trainer.train()