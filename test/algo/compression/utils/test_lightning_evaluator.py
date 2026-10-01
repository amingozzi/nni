# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

import nni
from nni.compression.utils.evaluator import LightningEvaluator


@nni.trace
class TinyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.dataset = TensorDataset(torch.randn(8, 4), torch.randn(8, 2))

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=2)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=2)


class TinyModule(pl.LightningModule):
    def __init__(self, configuration='optimizer'):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)
        self.training_steps = 0
        self.fail_train = False
        self.fail_test = False
        self.original_optimizer = nni.trace(torch.optim.SGD)(self.parameters(), lr=0.1)
        self.original_scheduler = nni.trace(torch.optim.lr_scheduler.ExponentialLR)(self.original_optimizer, 0.9)
        scheduler_config = {'scheduler': self.original_scheduler, 'interval': 'step', 'frequency': 2, 'name': 'rate'}
        self.optimizer_configuration = {
            'optimizer': self.original_optimizer,
            'dict_scheduler': {'optimizer': self.original_optimizer, 'lr_scheduler': self.original_scheduler},
            'dict_config': {'optimizer': self.original_optimizer, 'lr_scheduler': scheduler_config},
            'tuple_scheduler': ([self.original_optimizer], [self.original_scheduler]),
            'tuple_config': ([self.original_optimizer], [scheduler_config]),
            'list_dicts': [{'optimizer': self.original_optimizer, 'lr_scheduler': scheduler_config}],
        }[configuration]

    def forward(self, inputs):
        return self.layer(inputs)

    def training_step(self, batch, batch_idx):
        if self.fail_train:
            raise RuntimeError('Expected training failure')
        self.training_steps += 1
        inputs, targets = batch
        return torch.nn.functional.mse_loss(self(inputs), targets)

    def test_step(self, batch, batch_idx):
        if self.fail_test:
            raise RuntimeError('Expected evaluation failure')
        inputs, targets = batch
        self.log('default', torch.nn.functional.mse_loss(self(inputs), targets))

    def configure_optimizers(self):
        return self.optimizer_configuration


def create_evaluator(model, tmp_path):
    trainer = nni.trace(pl.Trainer)(
        max_epochs=4, max_steps=4, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path,
    )
    evaluator = LightningEvaluator(trainer, TinyDataModule())
    evaluator._init_optimizer_helpers(model)
    evaluator.bind_model(model)
    return evaluator


@pytest.mark.parametrize('configuration', [
    'optimizer', 'dict_scheduler', 'dict_config', 'tuple_scheduler', 'tuple_config', 'list_dicts',
])
def test_lightning_optimizer_configurations(configuration, tmp_path):
    model = TinyModule(configuration)
    evaluator = create_evaluator(model, tmp_path)
    original = model.optimizer_configuration
    if configuration.startswith('dict'):
        assert original['optimizer'] is model.original_optimizer
        scheduler = original['lr_scheduler']
        assert (scheduler['scheduler'] if isinstance(scheduler, dict) else scheduler) is model.original_scheduler
    elif configuration.startswith('tuple'):
        assert original[0][0] is model.original_optimizer
    elif configuration == 'list_dicts':
        assert original[0]['optimizer'] is model.original_optimizer

    reconstructed = model.configure_optimizers()
    if configuration in ('dict_config', 'list_dicts'):
        scheduler_config = reconstructed[0]['lr_scheduler']
    elif configuration == 'tuple_config':
        scheduler_config = reconstructed[1][0]
    else:
        scheduler_config = None
    if scheduler_config is not None:
        assert scheduler_config['interval'] == 'step'
        assert scheduler_config['frequency'] == 2
        assert scheduler_config['name'] == 'rate'

    evaluator.train(max_steps=2, max_epochs=1)
    assert model.training_steps == 2
    assert model._trainer is None
    assert evaluator.trainer.max_steps == 4
    assert evaluator.trainer.max_epochs == 4


@pytest.mark.parametrize('limits', [{'max_steps': 0}, {'max_epochs': 0}])
def test_lightning_zero_training_limits(limits, tmp_path):
    model = TinyModule()
    evaluator = create_evaluator(model, tmp_path)
    evaluator.train(**limits)
    assert model.training_steps == 0
    assert model._trainer is None


@pytest.mark.parametrize('operation', ['train', 'evaluate'])
def test_lightning_detaches_trainer_on_failure(operation, tmp_path):
    model = TinyModule()
    evaluator = create_evaluator(model, tmp_path)
    model.fail_train = operation == 'train'
    model.fail_test = operation == 'evaluate'
    with pytest.raises(RuntimeError, match='Expected .* failure'):
        getattr(evaluator, operation)()
    assert model._trainer is None


def test_lightning_optimizer_patches_revert_without_losing_parameters(tmp_path):
    model = TinyModule()
    evaluator = create_evaluator(model, tmp_path)
    events = []
    evaluator.patch_optimizer_step([lambda: events.append('before')], [lambda: events.append('after')])
    model.extra = torch.nn.Parameter(torch.randn(1))
    evaluator.patch_optim_param_group({'layer': [model.extra]})
    optimizer = model.configure_optimizers()[0]
    assert id(model.extra) in [id(param) for param in optimizer.param_groups[0]['params']]
    optimizer.step()
    assert events == ['before', 'after']

    evaluator.revert_optimizer_step()
    events.clear()
    optimizer.step()
    reconstructed = model.configure_optimizers()[0]
    assert id(model.extra) in [id(param) for param in reconstructed.param_groups[0]['params']]
    reconstructed.step()
    assert events == []


def test_lightning_rejects_empty_optimizer_configuration(tmp_path):
    model = TinyModule()
    model.optimizer_configuration = []
    with pytest.raises(ValueError, match='empty optimizer configuration'):
        create_evaluator(model, tmp_path)
