#!/usr/bin/env python
import os
import torch
from torch import nn

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from nnicotine.models import ResNet
from nnicotine.utils import get_data_loaders


log_interval = 5
batch_size = 1
data_root = os.path.join(os.environ["DATA_PATH"], "nnicotine")

model = ResNet()
train_loader, val_loader = get_data_loaders(data_root, batch_size=batch_size)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)

criterion = nn.CrossEntropyLoss(ignore_index=-1)

trainer = create_supervised_trainer(model, optimizer, criterion)

val_metrics = {"acc": Accuracy(), "loss": Loss(criterion)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics)

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(trainer):
    print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(trainer)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {} Acc: {:.2f} Loss: {:.2f}".format(trainer.state.epoch, metrics["acc"], metrics["loss"]))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {} Acc: {:.2f} Loss: {:.2f}".format(trainer.state.epoch, metrics["acc"], metrics["loss"]))

trainer.run(train_loader, max_epochs=100)
