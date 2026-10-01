# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import operator

import pytest
import torch

from nni.common.concrete_trace_utils import concrete_trace
from nni.common.concrete_trace_utils.concrete_tracer import node_is_impure_wrapper
from nni.common.graph_utils import TorchGraph
from nni.common.torch_utils import _temporary_eval_mode


class MixedModeModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.batchnorm = torch.nn.BatchNorm1d(4)
        self.batchnorm.eval()
        self.fail = fail

    def forward(self, inputs):
        if self.fail:
            raise RuntimeError('Expected tracing failure')
        return self.batchnorm(self.linear(inputs))


@pytest.mark.parametrize('trace', [concrete_trace, TorchGraph])
@pytest.mark.parametrize('fail', [False, True])
def test_tracing_preserves_module_modes(trace, fail):
    model = MixedModeModel(fail)
    modes = [module.training for module in model.modules()]
    inputs = torch.randn(8, 4)
    if fail:
        with pytest.raises(RuntimeError, match='Expected tracing failure'):
            trace(model, inputs if trace is TorchGraph else (inputs,))
    else:
        trace(model, inputs if trace is TorchGraph else (inputs,))
    assert [module.training for module in model.modules()] == modes


def test_temporary_eval_mode_failure():
    model = MixedModeModel()
    modes = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError, match='Expected failure'):
        with _temporary_eval_mode(model):
            assert not any(module.training for module in model.modules())
            raise RuntimeError('Expected failure')
    assert [module.training for module in model.modules()] == modes


def _tensor_function(inputs):
    return inputs.relu().add(1)


def test_concrete_trace_function_and_tensor_methods():
    inputs = torch.randn(8, 4)
    traced = concrete_trace(_tensor_function, (inputs,))
    torch.testing.assert_close(traced(inputs), _tensor_function(inputs))


def test_fx_impurity_arguments():
    graph = torch.fx.Graph()
    inputs = graph.placeholder('inputs')
    node = graph.call_function(operator.add, (inputs, 1))
    graph.output(node)
    assert node_is_impure_wrapper(node) == node.is_impure()
    assert node_is_impure_wrapper(node, False) == node.is_impure(False)
    assert node_is_impure_wrapper(node, impure_random=False) == node.is_impure(impure_random=False)
