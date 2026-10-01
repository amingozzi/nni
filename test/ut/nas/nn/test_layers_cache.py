# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import sys
import textwrap

import pytest


def test_valid_layer_cache_is_not_regenerated():
    script = """
        import importlib
        from unittest.mock import patch
        import nni.nas.nn.pytorch.layers as layers
        assert layers.validate_cache()
        with patch('inspect.getmembers', side_effect=AssertionError('Unexpected cache regeneration')):
            importlib.reload(layers)
    """
    subprocess.run([sys.executable, '-c', textwrap.dedent(script)], check=True)


def test_stale_readonly_cache_uses_generated_layers():
    script = """
        import importlib
        from pathlib import Path
        from unittest.mock import patch
        import pytest
        import nni.nas.nn.pytorch.layers as layers
        original_read_text = Path.read_text
        def read_text(path, *args, **kwargs):
            if path == layers.nn_cache_file_path:
                return '# _torch_version = invalid'
            return original_read_text(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read_text), patch('os.access', return_value=False):
            with pytest.warns(UserWarning, match='Cannot write'):
                importlib.reload(layers)
            assert layers.MutableLinear.__module__ == layers.__name__
    """
    subprocess.run([sys.executable, '-c', textwrap.dedent(script)], check=True)


@pytest.mark.parametrize('fail', [False, True])
def test_layer_cache_atomic_update(tmp_path, monkeypatch, fail):
    from pathlib import Path
    import nni.nas.nn.pytorch.layers as layers

    cache = tmp_path / '_layers.py'
    cache.write_text('old', encoding='utf-8')
    monkeypatch.setattr(layers, 'nn_cache_file_path', cache)
    original_replace = Path.replace

    def replace(source, target):
        assert cache.read_text(encoding='utf-8') == 'old'
        assert source.read_text(encoding='utf-8') == 'new'
        if fail:
            raise PermissionError('Expected cache write failure')
        return original_replace(source, target)

    monkeypatch.setattr(Path, 'replace', replace)
    if fail:
        with pytest.warns(RuntimeWarning, match='Failed to update'):
            assert not layers.write_cache('new')
    else:
        assert layers.write_cache('new')
    assert cache.read_text(encoding='utf-8') == ('old' if fail else 'new')
    assert list(tmp_path.iterdir()) == [cache]
