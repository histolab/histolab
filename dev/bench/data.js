window.BENCHMARK_DATA = {
  "lastUpdate": 1601470094161,
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
      },
      {
        "commit": {
          "author": {
            "email": "98marcolini@gmail.com",
            "name": "alessiamarcolini",
            "username": "alessiamarcolini"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "b6ab7858d36391092ecea69b2a765b6f657b99fd",
          "message": "Fix rgb_to_lab parameters",
          "timestamp": "2020-09-30T12:57:22+02:00",
          "tree_id": "d41d459f463145642e366bf9ed68808d63f05437",
          "url": "https://github.com/histolab/histolab/commit/b6ab7858d36391092ecea69b2a765b6f657b99fd"
        },
        "date": 1601463518276,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUnit::test_mask_difference",
            "value": 8.911278813556304,
            "unit": "iter/sec",
            "range": "stddev: 0.004205919284657756",
            "extra": "mean: 112.21733949999944 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUnit::test_mask_percent",
            "value": 222.40372340008216,
            "unit": "iter/sec",
            "range": "stddev: 0.0002706152879743462",
            "extra": "mean: 4.4963276006000115 msec\nrounds: 50"
          }
        ]
      },
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
          "id": "60f2f7bc89183f3b22e58046b1e1f3a5920a66f0",
          "message": "autocommit in benchmarks",
          "timestamp": "2020-09-30T14:46:05+02:00",
          "tree_id": "9f75a8b9d4bc1ebe73e15966405a195582089bb8",
          "url": "https://github.com/histolab/histolab/commit/60f2f7bc89183f3b22e58046b1e1f3a5920a66f0"
        },
        "date": 1601470092795,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUnit::test_mask_difference",
            "value": 8.127163018938408,
            "unit": "iter/sec",
            "range": "stddev: 0.005388445489934203",
            "extra": "mean: 123.04416654000164 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUnit::test_mask_percent",
            "value": 293.9198114664037,
            "unit": "iter/sec",
            "range": "stddev: 0.00041871956519548846",
            "extra": "mean: 3.4022885187999803 msec\nrounds: 50"
          }
        ]
      }
    ]
  }
}