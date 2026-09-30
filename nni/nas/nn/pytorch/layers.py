# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# FIXME: The dynamic generation causes issues upon un-install of NNI or
#        when NNI's been installed at a place with no write access.
#        We should find a better way to do this.

# If you've seen lint errors like `"Sequential" is not a known member of module`,
# please run `python test/vso_tools/trigger_import.py` to generate `_layers.py`.

import hashlib
import os
import tempfile
import warnings
from pathlib import Path

# To make auto-completion happy, we generate a _layers.py that lists out all the classes.
nn_cache_file_path = Path(__file__).parent / '_layers.py'

# Update this when cache format changes, to enforce an update.
cache_version = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:8]


def validate_cache() -> bool:
    import torch

    cache_valid = []

    if nn_cache_file_path.exists():
        lines = nn_cache_file_path.read_text().splitlines()
        for line in lines:
            if line.startswith('# _torch_version = '):
                _cached_torch_version = line[line.find('=') + 1:].strip()
                if _cached_torch_version == torch.__version__:
                    cache_valid.append(True)
            if line.startswith('# _torch_nn_cache_sha1 = '):
                _cached_cache_version = line[line.find('=') + 1:].strip()
                if _cached_cache_version == cache_version:
                    cache_valid.append(True)

    return len(cache_valid) >= 2 and all(cache_valid)


def generate_stub_file() -> str:
    import inspect

    import torch
    import torch.nn as nn

    _NO_WRAP_CLASSES = [
        # not an nn.Module
        'Parameter',
        'ParameterList',
        'Buffer',
        'UninitializedBuffer',
        'UninitializedParameter',
        'LinearCrossEntropyOptions',

        # arguments are special
        'Module',
        'Sequential',

        # utilities
        'Container',
        'DataParallel',
    ]

    _WRAP_WITHOUT_TAG_CLASSES = [
        # special support on graph engine
        'ModuleList',
        'ModuleDict',
    ]

    code = [
        '# Copyright (c) Microsoft Corporation.',
        '# Licensed under the MIT license.',
        '# This file is auto-generated to make auto-completion work.',
        '# When pytorch version does not match, it will get automatically updated.',
        '# pylint: skip-file',
        '# pyright: reportGeneralTypeIssues=false',
        f'# _torch_version = {torch.__version__}',
        f'# _torch_nn_cache_version = 10',  # backward compatibility
        f'# _torch_nn_cache_sha1 = {cache_version}',
        'import typing',
        'import torch.nn as nn',
        'from .base import ParametrizedModule',
    ]

    # Add modules, classes, functions in torch.nn into this module.
    for name, obj in inspect.getmembers(torch.nn):
        if inspect.isclass(obj):
            if name in _NO_WRAP_CLASSES:
                code.append(f'{name} = nn.{name}')
            elif not issubclass(obj, nn.Module):
                # It should never go here
                # We did it to play safe
                warnings.warn(f'{obj} is found to be not a nn.Module, which is unexpected. '
                              'It means your PyTorch version might not be supported.', RuntimeWarning)
                code.append(f'{name} = nn.{name}')
            elif name in _WRAP_WITHOUT_TAG_CLASSES:
                # for graph model space
                code.append(f'class {name}(ParametrizedModule, nn.{name}, wraps=nn.{name}, copy_wrapped=True):\n    _nni_basic_unit = False')  # pylint: disable=line-too-long
            else:
                code.append(f'class Mutable{name}(ParametrizedModule, nn.{name}, wraps=nn.{name}): pass')
                # for graph model space
                code.append(f'class {name}(ParametrizedModule, nn.{name}, wraps=nn.{name}, copy_wrapped=True): pass')

        elif inspect.isfunction(obj) or inspect.ismodule(obj):
            code.append(f'{name} = nn.{name}')  # no modification

    return '\n'.join(code)


def write_cache(code: str) -> bool:
    if os.access(nn_cache_file_path.as_posix(), os.W_OK) or (
        not nn_cache_file_path.exists() and os.access(nn_cache_file_path.parent.as_posix(), os.W_OK)
    ):
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=nn_cache_file_path.parent,
                prefix='_layers_', suffix='.tmp', delete=False
            ) as fp:
                temporary_path = Path(fp.name)
                fp.write(code)
            temporary_path.replace(nn_cache_file_path)
            return True
        except OSError as exc:
            warnings.warn(f'Failed to update {nn_cache_file_path}: {exc}', RuntimeWarning)
            return False
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    else:
        # no permission
        return False


if validate_cache():
    from ._layers import *  # pylint: disable=import-error, wildcard-import, unused-wildcard-import
else:
    code = generate_stub_file()
    if write_cache(code):
        from ._layers import *  # pylint: disable=import-error, wildcard-import, unused-wildcard-import
    else:
        warnings.warn(f'Cannot write to {nn_cache_file_path}. Will execute the generated code on-the-fly.')
        exec(code, globals())


def mutable_global_names():
    return [name for name, obj in globals().items() if isinstance(obj, type) and name.startswith('Mutable')]


# Export all the MutableXXX in this module by default.
__all__ = mutable_global_names()  # type: ignore
