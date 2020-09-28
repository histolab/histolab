import numpy as np
from histolab.filters.util import mask_percent, mask_difference


class TestDescribeBenchmarksFilterUnit:
    def test_mask_percent(self, benchmark):
        mask_array = np.random.choice(a=[False, True], size=(10000, 4000))
        benchmark.pedantic(mask_percent, args=[mask_array], iterations=100, rounds=50)

    def test_mask_difference(self, benchmark):
        minuend = np.random.choice(a=[False, True], size=(5000, 4000))
        subtrahend = np.random.choice(a=[False, True], size=(5000, 4000))
        benchmark.pedantic(mask_difference, args=[minuend, subtrahend], rounds=100)
