import tempfile
import pickle
import pytest
from pathlib import Path

import numpy as np
import torch

from nni.nas.utils import *

@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

def test_default_serializer():
    assert isinstance(get_default_serializer(), TorchSerializer)
    set_default_serializer(JsonSerializer())
    assert isinstance(get_default_serializer(), JsonSerializer)
    set_default_serializer(TorchSerializer())


def test_torch_serializer(tempdir, caplog):
    s = TorchSerializer()
    s.save(1, tempdir / 'test.ckpt')
    assert (tempdir / 'test.ckpt.torch').exists()

    assert s.load(tempdir / 'test.ckpt') == 1

    with pytest.raises(FileNotFoundError, match='No file found'):
        s.load(tempdir / 'test')
    assert 'does not match' in caplog.text

    caplog.clear()

    assert s.load(tempdir / 'test.ckpt.torch') == 1

    s.save(torch.randn(5), tempdir / 'test.ckpt')
    assert s.load(tempdir / 'test.ckpt').shape == (5,)


def test_json_serializer(tempdir, caplog):
    s = JsonSerializer()
    s.save({1: 5, 2: [3, 4]}, tempdir / 'test.ckpt')
    assert (tempdir / 'test.ckpt.json').exists()
    assert s.load(tempdir / 'test.ckpt') == {'1': 5, '2': [3, 4]}

    with pytest.raises(FileNotFoundError, match='No file found'):
        s.load(tempdir / 'test')
    assert 'does not match' in caplog.text


def test_torch_serializer_strategy_state(tempdir):
    random_state = np.random.RandomState(1).get_state()
    serializer = TorchSerializer(map_location='cpu')
    serializer.save({'random_state': random_state}, tempdir / 'strategy')
    restored = serializer.load(tempdir / 'strategy')['random_state']
    assert restored[0] == random_state[0]
    np.testing.assert_array_equal(restored[1], random_state[1])
    assert restored[2:] == random_state[2:]


def test_torch_serializer_weights_only(tempdir):
    serializer = TorchSerializer(weights_only=True)
    tensor = torch.randn(3)
    serializer.save({'weight': tensor}, tempdir / 'weights')
    torch.testing.assert_close(serializer.load(tempdir / 'weights')['weight'], tensor)

    serializer.save(np.random.RandomState(1).get_state(), tempdir / 'strategy')
    with pytest.raises(pickle.UnpicklingError, match='Weights only load failed'):
        serializer.load(tempdir / 'strategy')


def test_serializer_suffix_lookup_without_directory_scan(tempdir, monkeypatch):
    serializer = TorchSerializer()
    serializer.save(1, tempdir / 'checkpoint')

    def unexpected_scan(self):
        raise AssertionError('Existing checkpoints must not scan their parent directory.')

    monkeypatch.setattr(Path, 'iterdir', unexpected_scan)
    assert serializer.load(tempdir / 'checkpoint') == 1


def test_serializer_missing_parent(tempdir):
    with pytest.raises(FileNotFoundError, match='No file found'):
        TorchSerializer().load(tempdir / 'missing' / 'checkpoint')


def test_mixed_serializer(tempdir, caplog):
    s = TorchSerializer()
    s.save(1, tempdir / 'test.ckpt')

    s = JsonSerializer()
    with pytest.raises(FileNotFoundError, match='No file found'):
        s.load(tempdir / 'test.ckpt')

    assert 'which could be loaded' in caplog.text

    s.save(2, tempdir / 'test.ckpt')
    assert s.load(tempdir / 'test.ckpt') == 2

    assert TorchSerializer().load(tempdir / 'test.ckpt') == 1
