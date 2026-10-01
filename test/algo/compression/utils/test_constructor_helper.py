# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from nni.compression.utils.constructor_helper import OptimizerConstructHelper


def test_optimizer_parameter_order_and_groups():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    parameters = list(model.parameters())
    groups = [
        {'params': [parameters[3], parameters[1]], 'lr': 0.2},
        {'params': [parameters[2], parameters[0]], 'lr': 0.1},
    ]
    helper = OptimizerConstructHelper(model, torch.optim.SGD, groups, lr=0.01)
    optimizer = helper.call(model, None)

    assert helper.kwargs == {'lr': 0.01}
    for original, restored in zip(groups, optimizer.param_groups):
        assert [id(param) for param in original['params']] == [id(param) for param in restored['params']]
        assert original['lr'] == restored['lr']
        assert all(isinstance(param, torch.nn.Parameter) for param in original['params'])


def test_optimizer_parameter_name_mapping():
    model = torch.nn.Linear(3, 2)
    helper = OptimizerConstructHelper(model, torch.optim.SGD, [model.bias, model.weight], lr=0.1)
    wrapped = torch.nn.Sequential(model)
    optimizer = helper.call(wrapped, {'weight': '0.weight', 'bias': '0.bias'})
    assert [id(param) for param in optimizer.param_groups[0]['params']] == [id(model.bias), id(model.weight)]


def test_optimizer_rejects_unknown_parameters():
    model = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match='not part of the model'):
        OptimizerConstructHelper(model, torch.optim.SGD, [torch.nn.Parameter(torch.randn(2))], lr=0.1)

    helper = OptimizerConstructHelper(model, torch.optim.SGD, model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match='not part of the bound model'):
        helper.call(torch.nn.Linear(3, 2, bias=False), None)


def test_optimizer_indexes_model_parameters_once(monkeypatch):
    model = torch.nn.Linear(3, 2)
    calls = []
    original_named_parameters = model.named_parameters

    def named_parameters():
        calls.append(True)
        return original_named_parameters()

    monkeypatch.setattr(model, 'named_parameters', named_parameters)
    helper = OptimizerConstructHelper(
        model, torch.optim.SGD, [{'params': [model.bias]}, {'params': [model.weight]}], lr=0.1
    )
    assert len(calls) == 1
    helper.call(model, None)
    assert len(calls) == 2
