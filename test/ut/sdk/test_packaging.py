# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def setup_module(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location('nni_setup', root / 'setup.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_discovery(setup_module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for directory in (
        'nni/common',
        'nni/__pycache__/nested',
        'nni/common/.mypy_cache/nested',
        'nni/runtime/default_config',
        'nni_assets',
    ):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)

    assert setup_module._find_python_packages() == ['nni', 'nni.common', 'nni.runtime', 'nni_assets']

    (tmp_path / 'nni_node').mkdir()
    assert 'nni_node' in setup_module._find_python_packages()


def test_modern_framework_metadata(setup_module, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    metadata = {}
    monkeypatch.setattr(setup_module.setuptools, 'setup', lambda **kwargs: metadata.update(kwargs))
    setup_module._setup()
    assert metadata['python_requires'] == '>=3.10'
    assert metadata['package_dir'] == {'': '.'}
    assert 'torch>=2.6,<3' in metadata['extras_require']['nas']
    assert 'pytorch-lightning>=2.6,<3' in metadata['extras_require']['nas']
    assert 'torch>=2.6,<3' in metadata['extras_require']['compression']
    assert 'setup_requires' not in metadata


def test_setup_preserves_setuptools_distribution(setup_module):
    import distutils.core
    assert issubclass(distutils.core.Distribution, setup_module.setuptools.Distribution)


def test_global_node_toolchain(setup_module, monkeypatch, tmp_path):
    import setup_ts

    node = tmp_path / 'global_node'
    node.write_bytes(b'node runtime')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GLOBAL_TOOLCHAIN', '1')
    monkeypatch.setattr(setup_ts.shutil, 'which', lambda name: str(node) if name == 'node' else None)
    setup_ts.prepare_nni_node()
    assert (tmp_path / 'nni_node' / setup_ts.node_executable).read_bytes() == b'node runtime'


def test_missing_global_node_is_explicit(setup_module, monkeypatch, tmp_path):
    import setup_ts

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GLOBAL_TOOLCHAIN', '1')
    monkeypatch.setattr(setup_ts.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='requires node'):
        setup_ts.prepare_nni_node()


def test_node_bundle_preserves_dependency_lock(setup_module, monkeypatch, tmp_path):
    import json
    import setup_ts

    monkeypatch.chdir(tmp_path)
    manager = tmp_path / 'ts' / 'nni_manager'
    (manager / 'dist').mkdir(parents=True)
    (manager / 'dist' / 'main.js').write_text('', encoding='utf-8')
    (manager / 'dist' / 'nni_manager.tsbuildinfo').write_text('', encoding='utf-8')
    (manager / 'package.json').write_text(json.dumps({'name': 'nni', 'version': '999.0.0'}), encoding='utf-8')
    (manager / 'package-lock.json').write_text('{"lockfileVersion": 2}', encoding='utf-8')
    (tmp_path / 'ts' / 'webui' / 'build').mkdir(parents=True)
    (tmp_path / 'nni_node').mkdir()
    commands = []
    monkeypatch.setattr(setup_ts, '_npm', lambda path, *args: commands.append(args))
    setup_ts.copy_nni_node(None)
    assert (tmp_path / 'nni_node' / 'package-lock.json').read_text(encoding='utf-8') == '{"lockfileVersion": 2}'
    assert commands == [('ci', '--omit', 'dev')]
