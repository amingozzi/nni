# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any

from pytorch_lightning.utilities.combined_loader import (
    CombinedLoader, _CombinationMode, _SUPPORTED_MODES, _Sequential
)

_SUPPORTED_MODES['_nni_concat'] = _CombinationMode(fn=sum, iterator=_Sequential)

__all__ = ['ConcatLoader']


class ConcatLoader(CombinedLoader):
    """This is trying to bypass the supported mode checker in PyTorch-Lightning FitLoop.
    """

    def __init__(self, iterables: Any) -> None:
        super().__init__(iterables, mode='sequential')
        self._mode = '_nni_concat'

    def __len__(self) -> int:
        return len(_Sequential(self.flattened, self.limits))
