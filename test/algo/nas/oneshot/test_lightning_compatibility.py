# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from types import SimpleNamespace
from unittest.mock import Mock
import subprocess
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

from nni.nas.evaluator.pytorch import DataLoader, Regression, RegressionModule
from nni.nas.execution import SequentialExecutionEngine
from nni.nas.nn.pytorch import LayerChoice, ModelSpace
from nni.nas.oneshot.pytorch.base_lightning import BaseOneShotLightningModule
from nni.nas.space import RawFormatModelSpace
from nni.nas.strategy import DARTS, RandomOneShot


class TinySpace(ModelSpace):
    def __init__(self):
        super().__init__()
        self.choice = LayerChoice([torch.nn.Linear(4, 2), torch.nn.Linear(4, 2)], label='operation')

    def forward(self, inputs):
        return self.choice(inputs)


@pytest.mark.parametrize('first_import', [
    'nni.nas.oneshot.pytorch.base_lightning', 'nni.nas.strategy',
])
def test_strategy_import_order(first_import):
    script = (
        f'import {first_import}; '
        'from nni.nas import strategy; '
        'from nni.nas.oneshot.pytorch.strategy import DARTS, RandomOneShot; '
        'assert strategy.DARTS is DARTS; assert strategy.RandomOneShot is RandomOneShot'
    )
    subprocess.run([sys.executable, '-c', script], check=True)


@pytest.mark.parametrize('strategy_class', [DARTS, RandomOneShot])
def test_oneshot_training(strategy_class, tmp_path):
    torch.manual_seed(1)
    dataset = TensorDataset(torch.randn(8, 4), torch.randn(8, 2))
    evaluator = Regression(
        train_dataloaders=DataLoader(dataset, batch_size=2),
        val_dataloaders=DataLoader(dataset, batch_size=2),
        max_epochs=1, limit_train_batches=2, limit_val_batches=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=tmp_path, num_sanity_val_steps=0,
    )
    strategy = strategy_class()
    strategy(RawFormatModelSpace(TinySpace(), evaluator), SequentialExecutionEngine())
    assert len(list(strategy.list_models())) == 1
    assert evaluator.trainer.global_step >= 2


def create_scheduler_module():
    training_module = RegressionModule()
    module = BaseOneShotLightningModule(training_module)
    module.trainer = Mock(current_epoch=0)
    scheduler = SimpleNamespace(scheduler=Mock(), interval='epoch', frequency=1, reduce_on_plateau=False)
    module.trainer.lr_scheduler_configs = [scheduler]
    return module, training_module, scheduler


def test_scheduler_errors_are_not_hidden():
    module, training_module, _ = create_scheduler_module()
    training_module.lr_scheduler_step = Mock(side_effect=AttributeError('Expected scheduler failure'))
    with pytest.raises(AttributeError, match='Expected scheduler failure'):
        module._advance_lr_schedulers_impl(0, 'epoch')


def test_unsupported_plateau_scheduler_is_not_stepped():
    module, training_module, scheduler = create_scheduler_module()
    scheduler.reduce_on_plateau = True
    training_module.lr_scheduler_step = Mock()
    with pytest.warns(UserWarning, match='will be ignored'):
        module._advance_lr_schedulers_impl(0, 'epoch')
    training_module.lr_scheduler_step.assert_not_called()
