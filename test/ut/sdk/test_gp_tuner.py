# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from nni.algorithms.hpo.gp_tuner.target_space import TargetSpace


def test_gp_random_sample_uses_integer_scalars():
    space = TargetSpace({'integer': {'_type': 'randint', '_value': [-2, 3]}}, np.random.RandomState(1))
    for _ in range(10):
        sample = space.random_sample()
        assert sample.shape == (1,)
        assert -2 <= sample[0] < 3
        assert sample[0] == int(sample[0])
