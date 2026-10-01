# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING

from .base import Strategy
from .bruteforce import Random, GridSearch
from .evolution import RegularizedEvolution
from .hpo import TPEStrategy, TPE
from .rl import PolicyBasedRL

__all__ = [
    'Strategy', 'Random', 'GridSearch', 'RegularizedEvolution', 'TPEStrategy', 'TPE', 'PolicyBasedRL',
    'DARTS', 'Proxyless', 'GumbelDARTS', 'ENAS', 'RandomOneShot',
]

_oneshot_names = {'DARTS', 'Proxyless', 'GumbelDARTS', 'ENAS', 'RandomOneShot'}

if TYPE_CHECKING:
    from .oneshot import DARTS, Proxyless, GumbelDARTS, ENAS, RandomOneShot


def __getattr__(name):
    if name in _oneshot_names:
        from . import oneshot
        strategy = getattr(oneshot, name)
        globals()[name] = strategy
        return strategy
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
