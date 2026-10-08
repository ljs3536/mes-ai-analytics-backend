"""SSAD ProtoNet 학습. CNN 특징 추출기와 few-shot 기준점 학습이다."""

from __future__ import annotations

import random

import lightning.pytorch as pl
import numpy as np
import torch

from .dataset import KAERIBatchSampler, LeakDataset
from .loss import euclidean_dist
from .model import Classifier
from .taug import flip, jitter, magnitude_warp, scailing, time_warp, window_slice, window_warp


def _augment_list():
    return [
        (flip, 0.0, 1.0),
        (jitter, 0.0, 0.24),
        (scailing, 0.0, 0.8),
        (magnitude_warp, 0.0, 1.6),
        (time_warp, 0.0, 1.6),
        (window_slice, 0.0, 0.8),
        (window_warp, 0.0, 0.8),
    ]


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.augment_list = _augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 10) * float(maxval - minval) + minval
            img = op(img, val)
        return img


class BaseTrainer(pl.LightningModule):
    """정상 파형과 증강 파형의 임베딩 거리로 few-shot 기준점을 학습한다."""
    def __init__(self, n_filters, n_support, n_query, n, m, lr, normal_signals):
        super().__init__()
        self.save_hyperparameters(ignore=["normal_signals"])
        self.n_filters = n_filters
        self.n_support = n_support
        self.n_query = n_query
        self.lr = lr
        self.model = Classifier(n_filters=self.n_filters)
        self.transforms = RandAugment(n=n, m=m)
        self.normal_signals = np.asarray(normal_signals, dtype=np.float32)

    def train_dataloader(self):
        normal_data = self.normal_signals
        labels = np.array([1] * len(normal_data), dtype=int)
        labels = np.concatenate([labels, -labels])
        leak_data = normal_data.copy()
        data = np.concatenate([normal_data, leak_data])
        dataset = LeakDataset(data=data, labels=labels, transforms=self.transforms, normalize=False)
        sampler = KAERIBatchSampler(labels=labels, n_support=self.n_support, n_query=self.n_query, n_iters=20)
        return torch.utils.data.DataLoader(dataset, num_workers=0, batch_sampler=sampler, pin_memory=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=25)
        return [optimizer], [scheduler]

    def training_step(self, data, idx):
        signals, labels = data["data"], data["label"]
        logits = self.model(signals)
        support_logits = logits[: self.n_support]
        prototypes = support_logits.mean(0)
        query_logits = logits[self.n_support :]
        dists = euclidean_dist(query_logits, prototypes)
        query_labels = torch.LongTensor([0] * self.n_query + [1] * self.n_query).to(signals.device)
        loss = (dists ** (1 - 2 * query_labels)).sum()
        self.log("train_loss", loss, prog_bar=False, logger=False)
        return loss
