#!/usr/bin/env python3

import logging
import pickle

import jax
import jax.numpy as jnp
import flax.serialization
import numpy as np

from functools import partial
from operator import itemgetter
from os import rename
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Optional, Tuple

from argclass import Group, LogLevel, Parser
from flex.common import (Index, TrainState, accuracy, classify, loss_entropy,
                         make_rng)
from flex.tensorboard import SummaryWriter, tensorboard
from optax import piecewise_constant_schedule, add_decayed_weights, chain, sgd

# The source for it is in the folder nag_gs
from nag_gs import nag4, nag_gs

from resnet import ResNet20
from util import HessianWatcher


class HessianWatcherGroup(Group):

    watch: bool = False

    method: str = 'rayleigh'

    max_iters: int = 10

    step_size: float = 0.1


class Namespace(Parser):

    cache_dir: Path = 'cache'

    data_dir: Path = 'data'

    log_dir: Path = 'log'

    log_level: int = LogLevel

    alpha: float = 5e-2

    mu: float = 1.0

    gamma: float = 1.5

    learning_rate: float = 1e-1

    learning_schedule: str = 'constant'

    weight_decay: float = 1e-4

    batch_size: int = 256

    num_epoches: int = 100

    optimizer: str = 'sgd'

    seed: Optional[int] = None

    hessian = HessianWatcherGroup('hessian watch options')

    checkpoint: Optional[Path] = None


def load_cifar10(data_dir: Path, cache_dir: Optional[Path] = None,
                 reload: bool = False) -> Tuple[np.ndarray]:
    # Try to load from cache directory if reload is not forced.
    if reload:
        logging.info('ignore cached cifar-10 if exists')
    if cache_dir and cache_dir.exists():
        logging.info('load cached cifar-10')
        try:
            labels = np.fromfile(cache_dir / 'labels.bin', np.int32)
            images = np.fromfile(cache_dir / 'images.bin', np.float32)
            return images.reshape(-1, 32, 32, 3), labels
        except Exception as e:
            logging.error('failed to load cached cifar-10: %s', e)
            logging.info('fallback to reading raw cifar-10')

    # Read raw cifar-10 dataset and then apply some normalization.
    logging.info('read cifar-10 from %s', data_dir)
    if not (base_dir := data_dir / 'cifar-10-batches-py').exists():
        raise RuntimeError(f'Data directory does not exist: {base_dir}.')

    offset = 10000
    labels = np.empty(60000, np.int32)
    images = np.empty((labels.size, 32, 32, 3), np.float64)
    filenames = np.empty(labels.size, 'U40')

    def load_partition(path: Path, ix: slice):
        logging.info('load raw partition from %s', path)
        with open(path, 'rb') as fin:
            data = pickle.load(fin, encoding='latin1')
            filenames[ix] = data['filenames']
            labels[ix] = data['labels']
            images[ix, ...] = data['data'] \
                .astype(np.float64) \
                .reshape(-1, 3, 32, 32) \
                .transpose(0, 2, 3, 1)

    for i in range(0, 5):
        ix = slice(i * offset, (i + 1) * offset)
        path = base_dir / f'data_batch_{i + 1}'
        load_partition(path, ix)

    ix = slice((i + 1) * offset, (i + 2) * offset)
    path = base_dir / 'test_batch'
    load_partition(path, ix)

    # Normalize images with substracting mean and scaling with variance.
    #   mean:   [0.49186878, 0.48265391, 0.44717728]
    #   std:    [0.24697121, 0.24338894, 0.26159259]
    logging.info('estimate and apply normalization parameters to images')
    images /= 255
    images = (images - images.mean((0, 1, 2))) / images.std((0, 1, 2))
    images = images.astype(np.float32)

    # If there is a cache directory then we dump raw numpy array to files.
    if cache_dir:
        cache_dir.mkdir(exist_ok=True, parents=True)
        labels.tofile(cache_dir / 'labels.bin')
        images.tofile(cache_dir / 'images.bin')
        filenames.tofile(cache_dir / 'filenames.bin')

    return images, labels


def split_cifar10(dataset, val_size=5000, test_size=10000):
    total = 60000
    images, labels = dataset
    assert labels.shape == (total, )
    assert images.shape == (total, 32, 32, 3)

    if (train_size := total - (val_size + test_size)) <= 0:
        raise RuntimeError('There is nothing for train split.')

    logging.info('split cifar-10 dataset on %d train, %d val, and %d test',
                 train_size, val_size, test_size)

    train_ix = slice(train_size)
    train = (images[train_ix, ...], labels[train_ix])

    val_ix = slice(train_size, train_size + val_size)
    val = (images[val_ix, ...], labels[val_ix])

    test_ix = slice(train_size + val_size, total)
    test = (images[test_ix, ...], labels[test_ix])

    return train, val, test


def crop_jax(key, images):
    begins = jax.random.randint(key, (images.shape[0], 2), 0, 8)
    ends = begins + 32
    augmented = jnp.pad(images, [(0, 0), (4, 4), (4, 4), (0, 0)])
    cropped = jnp.zeros_like(images)
    for i, (begin, end) in enumerate(zip(begins, ends)):
        cropped = cropped \
            .at[i, ...] \
            .set(augmented[i, begin[0]:end[0], begin[1]:end[1]])
    return cropped


def crop_npy(images, rng=None):
    if not rng:
        rng = np.random.RandomState(42)
    begins = rng.randint(0, 8, (images.shape[0], 2))
    ends = begins + 32
    augmented = np.pad(images, [(0, 0), (4, 4), (4, 4), (0, 0)])
    cropped = np.zeros_like(images)
    for i, (begin, end) in enumerate(zip(begins, ends)):
        cropped[i, ...] = augmented[i, begin[0]:end[0], begin[1]:end[1]]
    return cropped


def make_crop_pt():
    import torch as T
    import torchvision

    crop = torchvision.transforms.RandomCrop(32, 2)

    def crop_fn(images):
        inp = T.tensor(images.transpose(0, 3, 1, 2), dtype=T.float32)
        out = crop(inp)
        return np.array(out).transpose(0, 2, 3, 1)

    return crop_fn


def flip_horizontal_jax_plain(key, images):
    mask_shape = images.shape[:1] + (1, 1, 1)
    mask = jax.random.randint(key, mask_shape, 0, 2).astype(bool)
    flipped = jnp.flip(images, 2)
    return jnp.where(mask, images, flipped)


flip_horizontal_jax = jax.jit(flip_horizontal_jax_plain)


def flip_horizontal_npy(images, rng=None, version=2):
    if not rng:
        rng = np.random.RandomState(42)
    if version == 1:
        mask = rng.randint(0, 2, images.shape[0], bool)
        flipped = np.empty_like(images)
        flipped[~mask] = images[~mask]
        flipped[mask] = np.flip(images[mask], 2)
        return flipped
    elif version == 2:
        mask_shape = images.shape[:1] + (1, 1, 1)
        mask = rng.randint(0, 2, mask_shape, bool)
        flipped = jnp.flip(images, 2)
        return np.where(mask, images, flipped)
    else:
        raise ValueError(f'Unknown implementation version: {version}.')


def augment(key, images, rng=None):
    cropped = crop_npy(images, rng=None)
    flipped = flip_horizontal_jax(key, cropped)
    return flipped


def dataset_batch(key, dataset, batch_size, collate_fn):
    """Function dataset_batch returns a single random batch from a dataset.
    """
    # Validate dataset shape.
    assert len(dataset) == 2
    images, labels = dataset
    assert images.shape[:1] == labels.shape

    # Generate permutation and slice images and labels.
    perms = jax.random.permutation(key, labels.size)
    perm = perms[:batch_size]
    inputs = jnp.array(collate_fn(images[perm, ...]))
    targets = jnp.array(labels[perm])
    return inputs, targets


def dataset_iterator(rng, dataset, batch_size, num_epoches: int = 1,
                     collate_fn: Optional[callable] = None):
    collate_fn = collate_fn or (lambda x: x)
    assert len(dataset) == 2
    images, labels = dataset
    assert images.shape[:1] == labels.shape

    num_samples = labels.size
    total_steps = num_samples // batch_size
    total_samples = total_steps * batch_size
    for i, key in enumerate(jax.random.split(rng, num_epoches)):
        perms = jax.random.permutation(key, num_samples)
        perms = perms[:total_samples]  # Skip incomplete batch.
        perms = perms.reshape((total_steps, batch_size))
        for j, perm in enumerate(perms):
            last_epoch = i + 1 == num_epoches
            last_batch = j + 1 == total_steps
            index = Index(i, j, i * total_steps + j, last_epoch, last_batch)
            yield index, collate_fn(images[perm, ...]), labels[perm, ...]


@jax.jit
def eval_batch(state: TrainState, xs, ys, rngs=None):
    # Apply model to eval batch.
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, xs, train=False)

    # Estimate target metrics on eval batch. Note, we aggregate over a batch.
    ranks = (1, 5)
    preds = classify(logits, (1, 5))
    scores = {f'acc@{rank}': accuracy(ys, x) for rank, x in zip(ranks, preds)}
    scores['loss'] = state.loss_fn(ys, logits, reduce=True)
    return scores


def make_eval_fn(dataset, rng=None):
    if rng is None:
        rng = make_rng()

    def eval_fn(state: TrainState, batch_size=256):
        dataset_it = dataset_iterator(rng, dataset, batch_size)
        for idx, xs, ys in dataset_it:
            if idx.step == 0:
                scores = eval_batch(state, xs, ys)
            else:
                score_batch = eval_batch(state, xs, ys)
                scores = jax.tree_map(lambda x, y: x + y, scores, score_batch)
        # Normalize on number of (full) batches.
        scores = jax.tree_map(lambda x: x / (idx.step + 1), scores)
        return scores

    return eval_fn


@jax.jit
def fit_batch(state: TrainState, xs, ys, rngs=None):
    rngs = jax.tree_map(jax.random.split, rngs or state.rngs)
    rngs_curr = jax.tree_map(itemgetter(0), rngs)
    rngs_next = jax.tree_map(itemgetter(1), rngs)

    @partial(jax.value_and_grad, has_aux=True)
    def objective(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        ps, aux = state.apply_fn(variables, xs, rngs=rngs_curr,
                                 mutable=['batch_stats'])
        ls = state.loss_fn(ys, ps)
        return ls, {'batch_stats': aux['batch_stats'], 'logits': ps}

    # Estimate loss gradients.
    (loss, aux), loss_grad = objective(state.params)

    # Make gradient step with gradient estimate.
    state = state \
        .apply_gradients(grads=loss_grad, batch_stats=aux['batch_stats']) \
        .replace(rngs=rngs_next)

    # Estimate target metrics on training batch.
    ranks = (1, 5)
    preds = classify(aux['logits'], (1, 5))
    scores = {f'acc@{rank}': accuracy(ys, x) for rank, x in zip(ranks, preds)}
    scores['loss'] = loss

    return state, scores


def fit(state: TrainState,
        dataset: Iterable[Tuple[Index, Any]],
        callback_fn: Optional[callable] = None,
        checkpoint: Optional[Path] = None,
        eval_fn: Optional[callable] = None,
        summary_writer: Optional[SummaryWriter] = None):

    def report_metrics(scores, scope=None):
        scope = scope + '/' if scope else ''
        metrics = ' '.join([
            f"{scope}loss={scores['loss']:e}",
            f"{scope}acc@1={scores['acc@1']:.2f}",
            f"{scope}acc@5={scores['acc@5']:.2f}",
        ])
        logging.info('[%3d:%6d] %s', idx.epoch, idx.step, metrics)
        for key, val in scores.items():
            summary_writer.add_scalar(f'{scope}{key}', val, state.step)

    rng = jax.random.PRNGKey(42)  # TODO
    score = float('inf')
    for idx, xs, ys in dataset:
        # On the first step we should initialize model and optimizer if they
        # are not initialized yet.
        if idx.step == 0:
            logging.info('initialize model on the first batch')
            state = state.init(rng, xs)

        # The essential part is to make step with stochastic gradient estimated
        # over batch.
        state, aux = fit_batch(state, xs, ys)

        # Save training state as soon as possible.
        if idx.last_batch and checkpoint:
            with NamedTemporaryFile(prefix='checkpoint-',
                                    suffix='.msgpack') as fout:
                fout.write(flax.serialization.to_bytes(state))
                rename(fout.name, checkpoint)

        # Report common training and test metrics.
        if idx.last_batch:
            report_metrics(aux, 'train')
        if idx.last_batch and eval_fn:
            scores = eval_fn(state)
            score = min(score, 1.0 - scores['acc@1'])
            report_metrics(scores, 'test')
            if np.isnan(scores.get('loss', float('nan'))):
                logging.info('loss on eval set is nan')
                return state, score

        # Do some work via callbacks if there are any.
        if idx.last_batch and callback_fn:
            callback_fn(state, idx, summary_writer)

    return state, score


def main(ns: Namespace):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=ns.log_level,
                        force=True)  # See google/jax #12374.

    boundaries = {32000: 0.1, 48000: 0.1}
    schedule = partial(piecewise_constant_schedule,
                       boundaries_and_scales=boundaries)

    if ns.optimizer == 'sgd':
        opt = chain(add_decayed_weights(ns.weight_decay),
                    sgd(schedule(ns.learning_rate), momentum=0.9))
    elif ns.optimizer == 'nag4':
        opt = nag4(alpha=schedule(ns.alpha), mu=ns.mu, gamma=ns.gamma)
    elif ns.optimizer == 'nag-gs':
        opt = nag_gs(alpha=schedule(ns.alpha), mu=ns.mu, gamma=ns.gamma)
    else:
        raise RuntimeError(f'Unknown optimizer: {ns.optimizer}.')

    key = jax.random.PRNGKey(42)
    rng = np.random.RandomState(42)
    collate_fn = partial(augment, key, rng=rng)
    dataset = load_cifar10(ns.data_dir, ns.cache_dir)
    trainset, _, testset = split_cifar10(dataset, val_size=0)
    trainset_it = dataset_iterator(key, trainset, ns.batch_size,
                                   ns.num_epoches, collate_fn)

    model = ResNet20()
    state = TrainState.create(apply_fn=model.apply,
                              predict_fn=classify,  # TODO Use everywhere.
                              init_fn=model.init,
                              loss_fn=loss_entropy,
                              tx=opt)

    eval_fn = make_eval_fn(testset)

    # Create callback function to watch on neural network hessian if it is
    # required with corresponding option.
    callback_fn = None
    if ns.hessian.watch:
        inputs, targets = dataset_batch(key, trainset, ns.batch_size,
                                        collate_fn)
        callback_fn = HessianWatcher(inputs, targets, ns.hessian.max_iters,
                                     ns.hessian.step_size)

    with tensorboard(str(ns.log_dir)) as summary_writer:
        _, score = fit(state=state,
                       dataset=trainset_it,
                       callback_fn=callback_fn,
                       checkpoint=ns.checkpoint,
                       eval_fn=eval_fn,
                       summary_writer=summary_writer)

    logging.info('done.')
    logging.info('best score is %f', score)
    return score


if __name__ == '__main__':
    main(Namespace(['resnet-cifar10.ini']).parse_args())
