# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections.abc import Iterator
from contextlib import contextmanager

from torch.nn import Module


@contextmanager
def _temporary_eval_mode(model: Module) -> Iterator[None]:
    training_states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in training_states:
            module.training = training
