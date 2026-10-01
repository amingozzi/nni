"""To test the cases of importing NAS without certain DL libraries installed."""

import argparse
import importlib.abc
import importlib.machinery
import subprocess
import sys

import pytest

masked_packages = ['torch', 'torch_none', 'tensorflow', 'tianshou']


class MissingDependency(importlib.abc.MetaPathFinder):
    def __init__(self, package):
        self.package = package

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.package or fullname.startswith(self.package + '.'):
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, path, target)


def mask_dependency(package):
    sys.meta_path = [
        MissingDependency(package) if finder is importlib.machinery.PathFinder else finder
        for finder in sys.meta_path
    ]


def import_related(mask_out):
    import nni
    nni.set_default_framework(mask_out)
    import nni.nas
    import nni.nas.evaluator
    import nni.nas.hub
    import nni.nas.strategy  # FIXME: this doesn't work yet
    import nni.nas.experiment


def import_rl_strategy_without_tianshou():
    from nni.nas.strategy import PolicyBasedRL
    with pytest.raises(ImportError, match='tianshou'):
        PolicyBasedRL()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('masked', choices=masked_packages)
    args = parser.parse_args()
    if args.masked == 'torch':
        mask_dependency('torch')
        import_related('tensorflow')
    elif args.masked == 'torch_none':
        mask_dependency('torch')
        import_related('none')
    elif args.masked == 'tensorflow':
        mask_dependency('tensorflow')
        import_related('pytorch')
    elif args.masked == 'tianshou':
        mask_dependency('tianshou')
        import_rl_strategy_without_tianshou()
    else:
        raise ValueError(f'Unknown masked package: {args.masked}')


@pytest.mark.parametrize('framework', masked_packages)
def test_import_without_framework(framework):
    subprocess.run([sys.executable, __file__, framework], check=True)


if __name__ == '__main__':
    main()
