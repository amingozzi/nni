# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy

import pytest
import torch

from nni.compression.speedup import ModelSpeedup


def test_speedup_restores_state_on_failure(monkeypatch):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(), torch.nn.Linear(4, 2)
    )
    model[1].eval()
    modes = [module.training for module in model.modules()]
    state = deepcopy(model.state_dict())
    graph = torch.fx.symbolic_trace(model)
    speedup = ModelSpeedup(model, torch.randn(8, 4), {}, graph_module=graph)
    assert [module.training for module in model.modules()] == modes
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value)

    def fail():
        with torch.no_grad():
            graph.get_submodule('0').weight.fill_(123)
        raise RuntimeError('Expected mask propagation failure')

    monkeypatch.setattr(speedup, 'fix_mask_conflict', fail)
    with pytest.raises(RuntimeError, match='Expected mask propagation failure'):
        speedup.speedup_model()
    assert [module.training for module in model.modules()] == modes
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value)


def test_speedup_compacts_linear_model():
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    reference = deepcopy(model).eval()
    weight_mask = torch.ones_like(model[0].weight)
    bias_mask = torch.ones_like(model[0].bias)
    weight_mask[2:] = 0
    bias_mask[2:] = 0
    with torch.no_grad():
        reference[0].weight.mul_(weight_mask)
        reference[0].bias.mul_(bias_mask)

    inputs = torch.randn(8, 4)
    masks = {'0': {'weight': weight_mask, 'bias': bias_mask}}
    ModelSpeedup(model, inputs, masks).speedup_model()
    assert model[0].out_features == 2
    assert model[2].in_features == 2
    torch.testing.assert_close(model(inputs), reference(inputs))
