window.BENCHMARK_DATA = {
  "lastUpdate": 1601335001504,
  "repoUrl": "https://github.com/histolab/histolab",
  "entries": {
    "Python Benchmark with pytest-benchmark": [
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "ernestoarbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "ernestoarbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "e99842d1442b8857d291ce5c934680d8bf3531e9",
          "message": "first working (hope) benchamrk workflow",
          "timestamp": "2020-09-29T01:00:19+02:00",
          "tree_id": "853497a7cf716b4b0ef517fe674b56d43b7c536b",
          "url": "https://github.com/histolab/histolab/commit/e99842d1442b8857d291ce5c934680d8bf3531e9"
        },
        "date": 1601335000735,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUnit::test_mask_difference",
            "value": 8.63375958556944,
            "unit": "iter/sec",
            "range": "stddev: 0.009283482386933049",
            "extra": "mean: 115.82439725000114 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUnit::test_mask_percent",
            "value": 223.96945903720888,
            "unit": "iter/sec",
            "range": "stddev: 0.000266531061133247",
            "extra": "mean: 4.464894473999984 msec\nrounds: 50"
          }
        ]
      }
    ]
  }
}