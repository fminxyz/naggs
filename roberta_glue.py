# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Fine-tuning RoBERTa on GLUE

# Based on HuggingFace's tutorial on ["Fune-tuning on Classification Tasks"][1]
# and [pre-trained RoBERTa][2] model.
#
# [1]: https://huggingface.co/docs/transformers/notebooks
# [2]: https://huggingface.co/roberta-base

# + [markdown] id="kTCFado4IrIc"
# The GLUE Benchmark is a group of nine classification tasks on sentences or
# pairs of sentences which are
#
# - [CoLA][1] (abbrv. _Corpus of Linguistic Acceptability_) Determine if a
#   sentence is grammatically correct or not.is a  dataset containing sentences
#   labeled grammatically correct or not.
# - [MNLI][2] (abbrv. _Multi-Genre Natural Language Inference_) Determine if a
#   sentence entails, contradicts or is unrelated to a given hypothesis. (This
#   dataset has two versions, one with the validation and test set coming from
#   the same distribution, another called mismatched where the validation and
#   test use out-of-domain data.)
# - [MRPC][3] (abbrv. _Microsoft Research Paraphrase Corpus_) Determine if two
#   sentences are paraphrases from one another or not.
# - [QNLI][4] (abbrv. _Question-answering Natural Language Inference_)
#   Determine if the answer to a question is in the second sentence or not.
# - [QQP][5] (abbrv. _Quora Question Pairs2_) Determine if two questions are
#   semantically equivalent or not.
# - [RTE][6] (abbrv. _Recognizing Textual Entailment_) Determine if a sentence
#   entails a given hypothesis or not.
# - [SST-2][7] (abbrv. _Stanford Sentiment Treebank_) Determine if the sentence
#   has a positive or negative sentiment.
# - [STS-B][8] (abbrv. _Semantic Textual Similarity Benchmark_) Determine the
#   similarity of two sentences with a score from 1 to 5.
# - [WNLI][9] (abbrv. _Winograd Natural Language Inference_) Determine if a
#   sentence with an anonymous pronoun and a sentence with this pronoun
#   replaced are entailed or not.
#
# [1]: https://nyu-mll.github.io/CoLA/
# [2]: https://arxiv.org/abs/1704.05426
# [3]: https://www.microsoft.com/en-us/download/details.aspx?id=52398
# [4]: https://rajpurkar.github.io/SQuAD-explorer/
# [5]: https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs
# [6]: https://aclweb.org/aclwiki/Recognizing_Textual_Entailment
# [7]: https://nlp.stanford.edu/sentiment/index.html
# [8]: http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark
# [9]: https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html

import libcontext
from naggs import NagGS

import logging
import re

from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from os import environ, makedirs
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Force using of fixed CUDA device.
if 'CUDA_VISIBLE_DEVICES' not in environ:
    environ['CUDA_VISIBLE_DEVICES'] = '0'

from datasets import load_dataset, load_metric
from torch import manual_seed
from torch.utils.tensorboard import SummaryWriter
from transformers import get_constant_schedule
from transformers import (RobertaTokenizerFast as Tokenizer,
                          RobertaForSequenceClassification as Model,
                          Trainer, TrainerCallback, TrainingArguments)
from transformers.integrations import TensorBoardCallback



DEVICE = 'cuda'

SEED = 3407
TASK = 'cola'

CACHE_DIR = Path('data/cache').expanduser()
DATA_DIR = Path('data/huggingface')
LOG_DIR = Path('log')
MODEL_DIR = Path('model')

TASK_TO_KEYS = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mnli-mm': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}

TASK_TO_HYPERPARAMS = {
    'cola': (16, 1e-5),
    'mnli': (16, 1e-5),
    'mnli-mm': (16, 1e-5),
    'mrpc': (16, 1e-5),
    'qnli': (16, 1e-5),  # NOTE We use batch size 16 instead of 32.
    'qqp': (32, 1e-5),
    'rte': (16, 2e-5),
    'sst2': (32, 2e-5),
    'stsb': (16, 1e-5),
    'wnli': (32, 1e-5),
}

# +
parser = ArgumentParser()

parser.add_argument('-c', '--cache-dir',
                    default=CACHE_DIR,
                    type=Path,
                    help='Directory to cache or original dataset files.')

parser.add_argument('-d', '--data-dir',
                    default=DATA_DIR,
                    type=Path,
                    help='Directory to cache preprocessed dataset files.')

parser.add_argument('-l', '--log-dir',
                    default=LOG_DIR,
                    type=Path,
                    help='Directory for TensorBoard logs.')

parser.add_argument('-m', '--model-dir',
                    default=MODEL_DIR,
                    type=Path,
                    help='Directory to save checkpoint files.')

parser.add_argument('-L', '--learning-rate',
                    default=5e-2,
                    type=float,
                    help='Learning rate. If it is not given use standard .')

parser.add_argument('-s', '--seed',
                    default=SEED,
                    type=int,
                    help='Random seed for reproducibility.')

parser.add_argument('task',
                    default=TASK,
                    choices=sorted(TASK_TO_HYPERPARAMS),
                    nargs='?',
                    help='GLUE task to learn.')


OptimizerTy = Literal['adamw', 'naggs']


class HistoryCallback(TrainerCallback):

    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)
        self.history = defaultdict(list)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        for k, v in logs.items():
            if not self.pattern.match(k):
                continue
            if isinstance(v, (int, float)):
                self.history[k].append(v)
            else:
                logging.warning(
                    "Trainer is attempting to log a value of "
                    f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                    "This invocation of Tensorboard's writer.add_scalar() "
                    "is incorrect so we dropped this attribute.")


# +
def compute_metric(task, metric, inputs):
    predictions, references = inputs
    if task != 'stsb':
        predictions = predictions.argmax(axis=1)
    else:
        predictions = predictions[..., 0]
    return metric.compute(predictions=predictions, references=references)


def preprocess(tokenizer, lhs, rhs, sample):
    if rhs is None:
        args = (sample[lhs],)
    else:
        args = (sample[lhs], sample[rhs])
    return tokenizer(*args,
                     max_length=512,
                     padding=True,
                     truncation=True,
                     return_tensors='np')


def setup(task: str,
          cache_dir: Path = Path('cache'),
          data_dir: Path = Path('data'),
          model_dir: Path = Path('model'),
          callbacks: Optional[List[TrainerCallback]] = None,
          optimizer_ty: OptimizerTy = 'adamw',
          learning_rate: Optional[float] = None,
          alpha: float = 5e-2,
          gamma: float = 1.0,
          mu: float = 1.0,
          seed: Optional[int] = None):
    # Load and configure model output head.
    if task in ('mnli', 'mnli-mm'):
        num_labels = 3
    elif task == 'stsb':
        num_labels = 1
    else:
        num_labels = 2
    model_path = 'roberta-base'
    model = Model.from_pretrained(model_path, num_labels=num_labels)

    # Load tokenizer from checkpoint.
    tokenizer = Tokenizer.from_pretrained(model_path)

    # Make dataset preprocessor.
    keys = TASK_TO_KEYS[task]
    func = partial(preprocess, tokenizer, *keys)

    # Load and preprocess dataset.
    dataset_path = 'glue'
    dataset_name = 'mnli' if task == 'mnli-mm' else task
    dataset = load_dataset(dataset_path, dataset_name, cache_dir=str(data_dir))
    dataset_cache = {key: str(cache_dir / f'glue-{task}-{key}.arrow')
                     for key in dataset.keys()}
    dataset_encoded = dataset.map(func,
                                  batched=True,
                                  cache_file_names=dataset_cache)

    # Load dataset metric.
    metric = load_metric(dataset_path, dataset_name)
    metric_compute = partial(compute_metric, task, metric)

    # Pick right dataset for train/evaluation stage.
    dataset_train = dataset_encoded['train']
    dataset_eval = dataset_encoded.get('validation')
    if task == 'mnli-mm':
        dataset_eval = dataset_encoded['validation_mismatched']
    elif task == 'mnli':
        dataset_eval = dataset_encoded['validation_matched']

    # Get reference hyperparameters from task name.
    bs, lr = TASK_TO_HYPERPARAMS[task]
    noepoches = 10

    # Configure either custom optimizer (via Trainer object) or default one
    # (via TrainingArgs object).
    trainer_kwargs = {}
    training_args_kwargs = {}

    if optimizer_ty == 'adamw':
        # Make 6% of total steps as warm up steps.
        warmup_steps = int(0.06 * len(dataset_train) * noepoches / bs)

        logging.info('use default optimizer: AdamW')
        args_kwargs = dict(
            learning_rate=learning_rate or lr,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-6,
            lr_scheduler_type='polynomial',
            warmup_steps=warmup_steps,
        )
    elif optimizer_ty == 'naggs':
        logging.info('use custom optimizer: NAG-GS')
        optimizer = NagGS(params=model.parameters(),
                         lr=alpha,
                         mu=mu,
                         gamma=gamma)
        scheduler = get_constant_schedule(optimizer)
        trainer_kwargs = {'optimizers': (optimizer, scheduler)}
    else:
        raise ValueError(f'Unknown optimizer type: {optimizer_ty}.')

    # Initialize training driver.
    args = TrainingArguments(output_dir=str(model_dir / f'glue-{task}'),
                             evaluation_strategy='epoch',
                             per_device_train_batch_size=bs,
                             per_device_eval_batch_size=bs,
                             num_train_epochs=noepoches,
                             save_strategy='no',
                             logging_strategy='epoch',
                             log_level='warning',
                             push_to_hub=False,
                             seed=seed,
                             data_seed=seed,
                             **training_args_kwargs)

    trainer = Trainer(model=model.to(DEVICE),
                      args=args,
                      train_dataset=dataset_train,
                      eval_dataset=dataset_eval,
                      tokenizer=tokenizer,
                      compute_metrics=metric_compute,
                      callbacks=callbacks,
                      **trainer_kwargs)

    return trainer


# -

def train(task: str, cache_dir: Path, data_dir: Path, log_dir: Path,
          model_dir: Path, learning_rate: Optional[float], seed: int,
          alpha: float, gamma: float = 1.0, mu: float = 1.0,
          optimizer_ty: OptimizerTy = 'adamw') -> Dict[str, List[Any]]:
    makedirs(cache_dir, exist_ok=True)
    makedirs(log_dir, exist_ok=True)
    makedirs(model_dir, exist_ok=True)

    manual_seed(seed)

    if optimizer_ty == 'adamw':
        lr_dir = 'default' if learning_rate is None else f'{learning_rate:e}'
        summary_dir = log_dir / 'baseline' / task / str(seed) / lr_dir
    elif optimizer_ty == 'naggs':
        summary_dir = (log_dir / 'naggs' / task / str(seed) /
                       f'{alpha:e}' / f'{mu:e}' / f'{gamma:e}')
    else:
        raise ValueError(f'Unknown optimizer type: {optimizer_ty}.')

    history_cb = HistoryCallback(r'eval_(accuracy|matthews_correlation'
                                 r'|pearson|loss)')
    tensorboard_sm = SummaryWriter(summary_dir)
    tensorboard_cb = TensorBoardCallback(tensorboard_sm)

    trainer = setup(task, cache_dir, data_dir, model_dir,
                    [history_cb, tensorboard_cb], optimizer_ty, learning_rate,
                    alpha, gamma, mu, seed)
    trainer.train()

    tensorboard_sm.flush()
    tensorboard_sm.close()

    return history_cb.history
