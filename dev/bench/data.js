window.BENCHMARK_DATA = {
  "lastUpdate": 1611093146595,
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "4aef626f6cf0ff9102e63cd9ebb46f268990eae7",
          "message": "new tests in benchmarks",
          "timestamp": "2020-09-30T16:07:25+02:00",
          "tree_id": "c991e4ef979b01aacedac8fae356b42b9acf4002",
          "url": "https://github.com/histolab/histolab/commit/4aef626f6cf0ff9102e63cd9ebb46f268990eae7"
        },
        "date": 1601474990540,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.784657664037161,
            "unit": "iter/sec",
            "range": "stddev: 0.0032356764650666213",
            "extra": "mean: 128.45780035000217 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 282.80847866122366,
            "unit": "iter/sec",
            "range": "stddev: 0.00007661560017624431",
            "extra": "mean: 3.5359618803999866 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1057.3147807011044,
            "unit": "iter/sec",
            "range": "stddev: 0.000015491014096698318",
            "extra": "mean: 945.7921313999805 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 35540.78355992677,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023983710293687287",
            "extra": "mean: 28.136689736000335 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 38585.95187038198,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034593607554622635",
            "extra": "mean: 25.916167712000515 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 32567.141446622198,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013533599994758278",
            "extra": "mean: 30.705795952002976 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "b05d520532654713aeaf453578689537c309b7bf",
          "message": "add python38 in setup.py",
          "timestamp": "2020-09-30T23:40:08+02:00",
          "tree_id": "2eb9538dff2b8d08fe138ac1d215330fe494b6e1",
          "url": "https://github.com/histolab/histolab/commit/b05d520532654713aeaf453578689537c309b7bf"
        },
        "date": 1601502122570,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.793908574484398,
            "unit": "iter/sec",
            "range": "stddev: 0.005799402696930317",
            "extra": "mean: 113.7150780600004 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 360.8863541592808,
            "unit": "iter/sec",
            "range": "stddev: 0.0003538371511343212",
            "extra": "mean: 2.770955422599991 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1011.7981068329487,
            "unit": "iter/sec",
            "range": "stddev: 0.00002779644096004124",
            "extra": "mean: 988.3394653999915 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 24731.238347060487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021365109741397",
            "extra": "mean: 40.43469178399869 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 26492.339280507156,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021077546283911002",
            "extra": "mean: 37.74676103200113 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24700.70715330469,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029867064922747046",
            "extra": "mean: 40.48467089599944 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fbd91e2a8143eb6cf8702b223167d4ef5fc79829",
          "message": "Update README.md",
          "timestamp": "2020-10-01T00:01:18+02:00",
          "tree_id": "dcc5b26a89e21791ca6429318ea5fb4103be67ed",
          "url": "https://github.com/histolab/histolab/commit/fbd91e2a8143eb6cf8702b223167d4ef5fc79829"
        },
        "date": 1601503388945,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.321684399568948,
            "unit": "iter/sec",
            "range": "stddev: 0.011143008392168093",
            "extra": "mean: 120.16797945999954 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 217.44862350244378,
            "unit": "iter/sec",
            "range": "stddev: 0.00024022491488789462",
            "extra": "mean: 4.598787446400007 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1045.1633586671269,
            "unit": "iter/sec",
            "range": "stddev: 0.00006745854073656738",
            "extra": "mean: 956.7882300000235 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 25241.20861415521,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032545764061965005",
            "extra": "mean: 39.61775425599876 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 25628.518566480787,
            "unit": "iter/sec",
            "range": "stddev: 0.000004125767323640795",
            "extra": "mean: 39.019032543999174 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 22978.055961152077,
            "unit": "iter/sec",
            "range": "stddev: 0.0000044390574415870325",
            "extra": "mean: 43.519782600001236 usec\nrounds: 250"
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
          "id": "7cc6af595056234a011586c823cf78f18601c509",
          "message": "Fix typo in quickstart",
          "timestamp": "2020-10-01T21:04:21+02:00",
          "tree_id": "b3ba5d2c2988078943cde386a1fad87d8cab60b7",
          "url": "https://github.com/histolab/histolab/commit/7cc6af595056234a011586c823cf78f18601c509"
        },
        "date": 1601579169049,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.4807194006586215,
            "unit": "iter/sec",
            "range": "stddev: 0.001423610535210892",
            "extra": "mean: 133.67698298000022 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 286.09654191968593,
            "unit": "iter/sec",
            "range": "stddev: 0.000042008363605420724",
            "extra": "mean: 3.4953236179999805 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1064.8760841759254,
            "unit": "iter/sec",
            "range": "stddev: 0.000004936417200379685",
            "extra": "mean: 939.0764004000232 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 34127.29156664356,
            "unit": "iter/sec",
            "range": "stddev: 1.3436817910249868e-7",
            "extra": "mean: 29.30206160800094 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 37653.8902747506,
            "unit": "iter/sec",
            "range": "stddev: 1.2497878968477198e-7",
            "extra": "mean: 26.557680832000642 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30399.17164445862,
            "unit": "iter/sec",
            "range": "stddev: 1.5009881612971626e-7",
            "extra": "mean: 32.89563320000161 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "8c8eb9f9589870d0c8e49992b55f7c14ced7e8a6",
          "message": "try to fix travis failure on windows/macos pt2",
          "timestamp": "2020-10-14T11:41:46+02:00",
          "tree_id": "e0e68910e853c8371227d7ef2662d6690e124c83",
          "url": "https://github.com/histolab/histolab/commit/8c8eb9f9589870d0c8e49992b55f7c14ced7e8a6"
        },
        "date": 1602668653298,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.140211349103764,
            "unit": "iter/sec",
            "range": "stddev: 0.005032084110492282",
            "extra": "mean: 122.84693321999555 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 268.74681422218856,
            "unit": "iter/sec",
            "range": "stddev: 0.0003269018326964666",
            "extra": "mean: 3.7209743411999736 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 967.106256666149,
            "unit": "iter/sec",
            "range": "stddev: 0.000011060268284048942",
            "extra": "mean: 1.0340125432000036 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 19580.0189750921,
            "unit": "iter/sec",
            "range": "stddev: 0.000002238847818625082",
            "extra": "mean: 51.07247348800365 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 20960.673670418753,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011646496758590427",
            "extra": "mean: 47.708390279997275 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 18375.14751619734,
            "unit": "iter/sec",
            "range": "stddev: 0.000016039532863020805",
            "extra": "mean: 54.4213318080042 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "0c954059a1ee425908da444f04d07828b19b4a9e",
          "message": "Fix typos",
          "timestamp": "2020-10-14T12:22:47+02:00",
          "tree_id": "d8664f4eda21ada4a38343cc3ff1944370bf4220",
          "url": "https://github.com/histolab/histolab/commit/0c954059a1ee425908da444f04d07828b19b4a9e"
        },
        "date": 1602671114386,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.47222080066598,
            "unit": "iter/sec",
            "range": "stddev: 0.008763086074115134",
            "extra": "mean: 118.03280668999946 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 198.6022968064224,
            "unit": "iter/sec",
            "range": "stddev: 0.0009528316159200585",
            "extra": "mean: 5.035188495200032 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1163.8807946520672,
            "unit": "iter/sec",
            "range": "stddev: 0.000022996573628907983",
            "extra": "mean: 859.1945194000232 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 27329.285097050095,
            "unit": "iter/sec",
            "range": "stddev: 0.000002523534661851699",
            "extra": "mean: 36.59078517600665 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29418.837215856147,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020668432354305664",
            "extra": "mean: 33.99182614399933 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 25121.765852251312,
            "unit": "iter/sec",
            "range": "stddev: 0.000002245763341726139",
            "extra": "mean: 39.80611896000073 usec\nrounds: 250"
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
          "id": "a0e4a95028289d27265b6ef37bc98f9524947795",
          "message": "release: prepare v0.1.1 release",
          "timestamp": "2020-10-14T21:49:09+02:00",
          "tree_id": "661956862ba3b097b73ee236eb306b8cc1548b4f",
          "url": "https://github.com/histolab/histolab/commit/a0e4a95028289d27265b6ef37bc98f9524947795"
        },
        "date": 1602705137130,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.050218145024633,
            "unit": "iter/sec",
            "range": "stddev: 0.01061634028757935",
            "extra": "mean: 124.22023626999987 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 210.7240091404988,
            "unit": "iter/sec",
            "range": "stddev: 0.00022580181525265732",
            "extra": "mean: 4.745543728400008 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1040.471236964586,
            "unit": "iter/sec",
            "range": "stddev: 0.00007250034482576267",
            "extra": "mean: 961.102973799973 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 24628.13970868702,
            "unit": "iter/sec",
            "range": "stddev: 0.000004780572279151408",
            "extra": "mean: 40.60396001600043 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 25493.162370828057,
            "unit": "iter/sec",
            "range": "stddev: 0.000004581282538168819",
            "extra": "mean: 39.22620448000225 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 22662.63409858544,
            "unit": "iter/sec",
            "range": "stddev: 0.0000068900262527403526",
            "extra": "mean: 44.125497312001244 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "7cf938fa56a4ad868c788ad4c3d61f723bb5d586",
          "message": "make __init__ for HistolabException consistent",
          "timestamp": "2020-10-28T22:36:16+01:00",
          "tree_id": "3ca40591bafc01f1c8f9f4463f870fb868ee90ff",
          "url": "https://github.com/histolab/histolab/commit/7cf938fa56a4ad868c788ad4c3d61f723bb5d586"
        },
        "date": 1603922625581,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.87816725540527,
            "unit": "iter/sec",
            "range": "stddev: 0.0025245612444240446",
            "extra": "mean: 145.38756661000662 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 181.66701488502696,
            "unit": "iter/sec",
            "range": "stddev: 0.00015329731636130634",
            "extra": "mean: 5.50457660480015 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1050.4562589638447,
            "unit": "iter/sec",
            "range": "stddev: 0.00001161155467130267",
            "extra": "mean: 951.9672917998378 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 33401.1080561065,
            "unit": "iter/sec",
            "range": "stddev: 3.8066128366170697e-7",
            "extra": "mean: 29.939126520001082 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 36508.47236098712,
            "unit": "iter/sec",
            "range": "stddev: 2.9751242422493434e-7",
            "extra": "mean: 27.390902311995887 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30048.933707760258,
            "unit": "iter/sec",
            "range": "stddev: 4.7448725204285414e-7",
            "extra": "mean: 33.279051087984044 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "77f06cf8998c0c1580b15be96e694603f9a70c60",
          "message": "fix unconsistent path method",
          "timestamp": "2020-10-29T00:11:25+01:00",
          "tree_id": "f765ed4c3ee758def664fbf1334fb52fd81c281d",
          "url": "https://github.com/histolab/histolab/commit/77f06cf8998c0c1580b15be96e694603f9a70c60"
        },
        "date": 1603928226696,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.326183914412183,
            "unit": "iter/sec",
            "range": "stddev: 0.0018072901260508308",
            "extra": "mean: 136.49670984000068 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 283.93514499677923,
            "unit": "iter/sec",
            "range": "stddev: 0.0000601036364047141",
            "extra": "mean: 3.5219310382000915 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1049.0449574018946,
            "unit": "iter/sec",
            "range": "stddev: 0.00000783699169626119",
            "extra": "mean: 953.2479927996974 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 34177.71105639406,
            "unit": "iter/sec",
            "range": "stddev: 5.293232203382339e-7",
            "extra": "mean: 29.25883475198134 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 37760.72261738912,
            "unit": "iter/sec",
            "range": "stddev: 4.979851547344637e-7",
            "extra": "mean: 26.4825440480181 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30813.30569695803,
            "unit": "iter/sec",
            "range": "stddev: 6.228791116561198e-7",
            "extra": "mean: 32.45351244799167 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "9ec45ac85674bc7aca358e2b0b3b7670252bee8b",
          "message": "Fix quickstart path",
          "timestamp": "2020-10-30T10:59:03+01:00",
          "tree_id": "842a8794eb6094791ab5741aa3d3042d63a94610",
          "url": "https://github.com/histolab/histolab/commit/9ec45ac85674bc7aca358e2b0b3b7670252bee8b"
        },
        "date": 1604053079730,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.661793406514203,
            "unit": "iter/sec",
            "range": "stddev: 0.0060512925121707735",
            "extra": "mean: 115.44953257000316 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 316.3833085825984,
            "unit": "iter/sec",
            "range": "stddev: 0.0000359702647491815",
            "extra": "mean: 3.16072299919997 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1097.562840770852,
            "unit": "iter/sec",
            "range": "stddev: 0.00003544335852103817",
            "extra": "mean: 911.1095627997655 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 41028.57431877472,
            "unit": "iter/sec",
            "range": "stddev: 0.000002066167323370058",
            "extra": "mean: 24.373257335983 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 45558.04564736472,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018833032872374138",
            "extra": "mean: 21.950019712003268 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 37101.988360497424,
            "unit": "iter/sec",
            "range": "stddev: 0.000002504383745638349",
            "extra": "mean: 26.95273337600156 usec\nrounds: 250"
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
            "email": "31658006+nicolebussola@users.noreply.github.com",
            "name": "nicolebussola",
            "username": "nicolebussola"
          },
          "distinct": true,
          "id": "ea596047dc5e05ad18a524c1a9f8771122ae2473",
          "message": "Remove examples folder from pip installation",
          "timestamp": "2020-11-03T14:54:22+01:00",
          "tree_id": "b57701c1a102b40017d7db5626d0a734976202a5",
          "url": "https://github.com/histolab/histolab/commit/ea596047dc5e05ad18a524c1a9f8771122ae2473"
        },
        "date": 1604412961861,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.253591491613063,
            "unit": "iter/sec",
            "range": "stddev: 0.003190936757626622",
            "extra": "mean: 121.15937662000306 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 260.3230322547827,
            "unit": "iter/sec",
            "range": "stddev: 0.00015235720074921453",
            "extra": "mean: 3.841381192200015 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1130.9487081830757,
            "unit": "iter/sec",
            "range": "stddev: 0.00002633319913471695",
            "extra": "mean: 884.213397799931 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26757.66813456131,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012038351578491662",
            "extra": "mean: 37.37246440800118 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29264.87119292382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021782712241243552",
            "extra": "mean: 34.1706612480084 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24756.82286918292,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022695920552891422",
            "extra": "mean: 40.39290523198724 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicole.bussolaceradini@gmail.com",
            "name": "Nicole Bussola",
            "username": "nicolebussola"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "d9b310743865bf4df4d97d4b95d85b2552656c24",
          "message": "fix flake8",
          "timestamp": "2020-11-04T19:31:33+01:00",
          "tree_id": "3f7b8b92fbc606d5cb3f1dc4b516c3971e6c30ea",
          "url": "https://github.com/histolab/histolab/commit/d9b310743865bf4df4d97d4b95d85b2552656c24"
        },
        "date": 1604516015467,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.178993575437284,
            "unit": "iter/sec",
            "range": "stddev: 0.003925411828715287",
            "extra": "mean: 108.94440569998551 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 276.36417212022354,
            "unit": "iter/sec",
            "range": "stddev: 0.00011385978016318289",
            "extra": "mean: 3.6184140379997642 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1101.3105550891887,
            "unit": "iter/sec",
            "range": "stddev: 0.000022043768670423725",
            "extra": "mean: 908.0090946000382 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 27728.415566121443,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017246408670957061",
            "extra": "mean: 36.064087311999174 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29625.549607717505,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020239445560439835",
            "extra": "mean: 33.754648039997846 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24476.508181929246,
            "unit": "iter/sec",
            "range": "stddev: 0.0000042570018663963105",
            "extra": "mean: 40.85550081601468 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicole.bussolaceradini@gmail.com",
            "name": "Nicole Bussola",
            "username": "nicolebussola"
          },
          "committer": {
            "email": "31658006+nicolebussola@users.noreply.github.com",
            "name": "nicolebussola",
            "username": "nicolebussola"
          },
          "distinct": true,
          "id": "21dcf0697eef827268c928a7764faf6a816bf399",
          "message": "add warning for TCGA",
          "timestamp": "2020-11-05T18:30:55+01:00",
          "tree_id": "84a46dbecf8ee6201cba07d595b89a2a069bc0e0",
          "url": "https://github.com/histolab/histolab/commit/21dcf0697eef827268c928a7764faf6a816bf399"
        },
        "date": 1604598028854,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.424765301138402,
            "unit": "iter/sec",
            "range": "stddev: 0.006437825744833653",
            "extra": "mean: 118.69766862999427 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 280.3052759687505,
            "unit": "iter/sec",
            "range": "stddev: 0.000874651794467078",
            "extra": "mean: 3.5675389859999767 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 974.1801162685134,
            "unit": "iter/sec",
            "range": "stddev: 0.000029021417533702886",
            "extra": "mean: 1.0265042195999512 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 24535.059147244545,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016673668607904393",
            "extra": "mean: 40.75800241599609 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 27181.041602536985,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017065973245342158",
            "extra": "mean: 36.79034875200159 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24948.7574875261,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023044257760720035",
            "extra": "mean: 40.08215641600509 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "valerio.maggio@gmail.com",
            "name": "leriomaggio",
            "username": "leriomaggio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "8b07d870020e291cfd69b1d0c63905d43de577ba",
          "message": "Extra Reqs for examples with Instructions & README update",
          "timestamp": "2020-11-09T15:34:39+01:00",
          "tree_id": "36aa025c2262f2bb9bbc155d77558fa5bca009c7",
          "url": "https://github.com/histolab/histolab/commit/8b07d870020e291cfd69b1d0c63905d43de577ba"
        },
        "date": 1604933077316,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.08889978717645,
            "unit": "iter/sec",
            "range": "stddev: 0.004913659152026698",
            "extra": "mean: 110.0243179500012 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 309.36637315181144,
            "unit": "iter/sec",
            "range": "stddev: 0.00020586105746886352",
            "extra": "mean: 3.2324133674000906 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 991.2312755252661,
            "unit": "iter/sec",
            "range": "stddev: 0.000018461297314729854",
            "extra": "mean: 1.0088462951999646 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 22461.002560927067,
            "unit": "iter/sec",
            "range": "stddev: 0.000001980080581331713",
            "extra": "mean: 44.521610167998006 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 24676.145533140323,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018325176875994396",
            "extra": "mean: 40.52496767199682 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 23991.252842959024,
            "unit": "iter/sec",
            "range": "stddev: 0.000001931646128500328",
            "extra": "mean: 41.681858239991016 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "e72c529829e5c8b56b719624cd191abae9d32ee3",
          "message": "revert",
          "timestamp": "2020-11-14T20:52:31+01:00",
          "tree_id": "cf03efa272cc12f7e5a4454ecb4a881c73c5c951",
          "url": "https://github.com/histolab/histolab/commit/e72c529829e5c8b56b719624cd191abae9d32ee3"
        },
        "date": 1605384134555,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.591986708880312,
            "unit": "iter/sec",
            "range": "stddev: 0.009634311689251633",
            "extra": "mean: 151.69933498999058 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 198.74654847979627,
            "unit": "iter/sec",
            "range": "stddev: 0.0003564280033918983",
            "extra": "mean: 5.031533919199887 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 824.3113108815036,
            "unit": "iter/sec",
            "range": "stddev: 0.00009234038579290213",
            "extra": "mean: 1.2131339056000796 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 18418.711656754163,
            "unit": "iter/sec",
            "range": "stddev: 0.000008041460883913619",
            "extra": "mean: 54.29261387200768 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 19728.880079596154,
            "unit": "iter/sec",
            "range": "stddev: 0.000007280553784695774",
            "extra": "mean: 50.68711431999691 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16780.160852984078,
            "unit": "iter/sec",
            "range": "stddev: 0.000007889532613205794",
            "extra": "mean: 59.59418439199089 usec\nrounds: 250"
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
          "id": "9ad0045a4f8729600464c0152ec732a1c719aadb",
          "message": "fix master branch reference",
          "timestamp": "2020-11-14T20:56:45+01:00",
          "tree_id": "2fa88b9c00f14b73244244f804f69c6ebc399326",
          "url": "https://github.com/histolab/histolab/commit/9ad0045a4f8729600464c0152ec732a1c719aadb"
        },
        "date": 1605384185911,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.390785291950825,
            "unit": "iter/sec",
            "range": "stddev: 0.006159634156873237",
            "extra": "mean: 119.17835640000078 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 331.8423452347058,
            "unit": "iter/sec",
            "range": "stddev: 0.000026566251043909554",
            "extra": "mean: 3.0134791847999964 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1108.2319280099273,
            "unit": "iter/sec",
            "range": "stddev: 0.00002125649055705708",
            "extra": "mean: 902.3381972000379 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 40627.77670032183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016173821047976662",
            "extra": "mean: 24.61370227999896 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 45909.20342769731,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017575226958839195",
            "extra": "mean: 21.782124832004683 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 36570.50075500687,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020555589703198637",
            "extra": "mean: 27.34444372799817 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fef91d1f855e15432eb83398d578b7b5fade8ef7",
          "message": "Update README.md",
          "timestamp": "2020-11-14T21:05:44+01:00",
          "tree_id": "e878dbac0d113018303e6f0ea2daab94941a913a",
          "url": "https://github.com/histolab/histolab/commit/fef91d1f855e15432eb83398d578b7b5fade8ef7"
        },
        "date": 1605384778329,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.899045407381617,
            "unit": "iter/sec",
            "range": "stddev: 0.005035769742005306",
            "extra": "mean: 112.3716032699997 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 222.45483730357145,
            "unit": "iter/sec",
            "range": "stddev: 0.00015598904067797841",
            "extra": "mean: 4.495294470200065 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1078.8411956558377,
            "unit": "iter/sec",
            "range": "stddev: 0.000028133276205283935",
            "extra": "mean: 926.9204810000701 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 24208.63969072933,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035770951025045435",
            "extra": "mean: 41.30756675200337 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 26587.32750066302,
            "unit": "iter/sec",
            "range": "stddev: 0.000003266957613388984",
            "extra": "mean: 37.61190363999776 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 23082.5078051704,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036468610053040366",
            "extra": "mean: 43.32284899199749 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "a40b5bef25b9fa82cd4ffde813619a9b19f24ef9",
          "message": "Modify tests according to new exception",
          "timestamp": "2020-11-15T16:04:38+01:00",
          "tree_id": "a1995be36052817ff41589a3a241fb53f839b699",
          "url": "https://github.com/histolab/histolab/commit/a40b5bef25b9fa82cd4ffde813619a9b19f24ef9"
        },
        "date": 1605453100012,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.06852913990064,
            "unit": "iter/sec",
            "range": "stddev: 0.007782482773098027",
            "extra": "mean: 110.27146569999957 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 303.3975802907787,
            "unit": "iter/sec",
            "range": "stddev: 0.0005915953815393763",
            "extra": "mean: 3.296005192399991 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 976.2129395010152,
            "unit": "iter/sec",
            "range": "stddev: 0.000030308970473269128",
            "extra": "mean: 1.0243666720000077 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 23781.968887896244,
            "unit": "iter/sec",
            "range": "stddev: 0.000002017125415636339",
            "extra": "mean: 42.04866319999883 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 26352.07402721004,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021663555863096373",
            "extra": "mean: 37.94767724800113 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24123.21377386327,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027914475167726466",
            "extra": "mean: 41.45384646400089 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "2f72c501822a22946ca454be936e1445aecad7bb",
          "message": "chache jobs in gh action workflow",
          "timestamp": "2020-11-16T11:46:17+01:00",
          "tree_id": "c81185614781efd0137f43f80d7ab042c9fd4bb9",
          "url": "https://github.com/histolab/histolab/commit/2f72c501822a22946ca454be936e1445aecad7bb"
        },
        "date": 1605524079644,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.707295471793845,
            "unit": "iter/sec",
            "range": "stddev: 0.0023523893448119527",
            "extra": "mean: 149.0913892500032 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 196.14507152984316,
            "unit": "iter/sec",
            "range": "stddev: 0.0001232852058348428",
            "extra": "mean: 5.098267278399862 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1052.101893044112,
            "unit": "iter/sec",
            "range": "stddev: 0.000007289904086359044",
            "extra": "mean: 950.4782822000607 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 32629.86180082546,
            "unit": "iter/sec",
            "range": "stddev: 1.2421991051100661e-7",
            "extra": "mean: 30.64677399199718 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 36621.552449234216,
            "unit": "iter/sec",
            "range": "stddev: 4.0808994152889033e-7",
            "extra": "mean: 27.306324640011553 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30688.198891658267,
            "unit": "iter/sec",
            "range": "stddev: 4.1512501979599394e-7",
            "extra": "mean: 32.58581592000246 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "2e90a45e3b8a10c7b49f12b8cbe1eb5094feddad",
          "message": "typing __getitem__ in slide.py",
          "timestamp": "2020-11-22T22:27:31+01:00",
          "tree_id": "eeb680575ed9f336570ed9ed9f37c426479b3e40",
          "url": "https://github.com/histolab/histolab/commit/2e90a45e3b8a10c7b49f12b8cbe1eb5094feddad"
        },
        "date": 1606080928848,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.359078233227716,
            "unit": "iter/sec",
            "range": "stddev: 0.003186446285203609",
            "extra": "mean: 135.88658366000232 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 312.0177302243085,
            "unit": "iter/sec",
            "range": "stddev: 0.00003893748710891783",
            "extra": "mean: 3.2049460755999464 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1036.404983231639,
            "unit": "iter/sec",
            "range": "stddev: 0.000013308489693399",
            "extra": "mean: 964.873786000021 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 35208.46068795904,
            "unit": "iter/sec",
            "range": "stddev: 8.399633984635978e-7",
            "extra": "mean: 28.402264128008028 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 38621.50915651104,
            "unit": "iter/sec",
            "range": "stddev: 8.611772602093151e-7",
            "extra": "mean: 25.892307727996013 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 31995.15119372047,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012002114519041851",
            "extra": "mean: 31.254735879987496 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "9a3910a02106f6963729950f6da1056145e48822",
          "message": "Break when there is a flake8 error",
          "timestamp": "2020-11-22T23:18:38+01:00",
          "tree_id": "80c2838a81d7d59e2e03d2930949470a6db22015",
          "url": "https://github.com/histolab/histolab/commit/9a3910a02106f6963729950f6da1056145e48822"
        },
        "date": 1606084003182,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.911656461505606,
            "unit": "iter/sec",
            "range": "stddev: 0.00234235242391134",
            "extra": "mean: 144.6831169299992 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 300.61927218948534,
            "unit": "iter/sec",
            "range": "stddev: 0.00008420285090391306",
            "extra": "mean: 3.32646670559991 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1034.3711026828826,
            "unit": "iter/sec",
            "range": "stddev: 0.000009695264636273167",
            "extra": "mean: 966.7710142000942 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 33682.5901404449,
            "unit": "iter/sec",
            "range": "stddev: 1.3086094727343197e-7",
            "extra": "mean: 29.688928191993 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 36958.64375661513,
            "unit": "iter/sec",
            "range": "stddev: 3.6325160155303915e-7",
            "extra": "mean: 27.05726992000382 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30157.687568599904,
            "unit": "iter/sec",
            "range": "stddev: 9.555854044411675e-8",
            "extra": "mean: 33.1590410480012 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "9f8e1d9c4ff172086b8613567d9e46710e3aa242",
          "message": "address alessia's comments",
          "timestamp": "2020-11-27T10:25:05+01:00",
          "tree_id": "ab0e678e09f1f0b22265fa2599508dc83bb90ed8",
          "url": "https://github.com/histolab/histolab/commit/9f8e1d9c4ff172086b8613567d9e46710e3aa242"
        },
        "date": 1606469547358,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.464395594011736,
            "unit": "iter/sec",
            "range": "stddev: 0.0025939980975884756",
            "extra": "mean: 133.96931973999926 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 310.60345640174296,
            "unit": "iter/sec",
            "range": "stddev: 0.000011510406526700852",
            "extra": "mean: 3.2195391885999243 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1073.2335380459579,
            "unit": "iter/sec",
            "range": "stddev: 0.000008248344488889173",
            "extra": "mean: 931.7636512000036 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 36509.326749719316,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010362901907334748",
            "extra": "mean: 27.390261311999893 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 39987.13609353268,
            "unit": "iter/sec",
            "range": "stddev: 9.433530458460065e-7",
            "extra": "mean: 25.00804252800026 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 32990.5116746914,
            "unit": "iter/sec",
            "range": "stddev: 0.000001020515915721386",
            "extra": "mean: 30.311745687992698 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "7feca1d51281a1a059ace29d9cce6cafcffe8764",
          "message": "mv markdown table to html",
          "timestamp": "2020-11-27T19:14:59+01:00",
          "tree_id": "67bd63cb731a9a3a2bf08cb45ffede1c27d178e8",
          "url": "https://github.com/histolab/histolab/commit/7feca1d51281a1a059ace29d9cce6cafcffe8764"
        },
        "date": 1606501391124,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.088659434988834,
            "unit": "iter/sec",
            "range": "stddev: 0.002545598131084764",
            "extra": "mean: 123.62988058000496 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 239.770766774793,
            "unit": "iter/sec",
            "range": "stddev: 0.00014955263386599065",
            "extra": "mean: 4.170650214999978 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1057.6322816909174,
            "unit": "iter/sec",
            "range": "stddev: 0.00002397261582860702",
            "extra": "mean: 945.5082047998985 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 23730.012722613686,
            "unit": "iter/sec",
            "range": "stddev: 0.000003367060062886562",
            "extra": "mean: 42.14072751200183 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 26425.380718023272,
            "unit": "iter/sec",
            "range": "stddev: 0.000003596103939956809",
            "extra": "mean: 37.842406535999544 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 22124.939379037278,
            "unit": "iter/sec",
            "range": "stddev: 0.000003514695240478896",
            "extra": "mean: 45.19786395200117 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "888766ddd56b64ab95a8d36b39ba828ed511d8c5",
          "message": "workflow for pypi release",
          "timestamp": "2020-12-04T17:31:30+01:00",
          "tree_id": "5f0a458ed2c8e3097eb5716693dd6adad08122dd",
          "url": "https://github.com/histolab/histolab/commit/888766ddd56b64ab95a8d36b39ba828ed511d8c5"
        },
        "date": 1607099940506,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.118037614384024,
            "unit": "iter/sec",
            "range": "stddev: 0.0033327106490046083",
            "extra": "mean: 109.6727215099952 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 266.17989151739584,
            "unit": "iter/sec",
            "range": "stddev: 0.00011537866467375063",
            "extra": "mean: 3.7568577938001226 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1121.1076033330821,
            "unit": "iter/sec",
            "range": "stddev: 0.00002152725884849233",
            "extra": "mean: 891.9750406000048 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26293.45008797527,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013624188735006844",
            "extra": "mean: 38.0322854799997 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29169.44195529821,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012707183547576412",
            "extra": "mean: 34.28245221600355 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24940.117466115425,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017865977673262714",
            "extra": "mean: 40.096042103997206 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "72a598eb0e35536581ed1227e0ebacc57a17e915",
          "message": "remove code inspector",
          "timestamp": "2020-12-04T18:25:18+01:00",
          "tree_id": "b3e8447d95512249d56f95f59788a09badb35f62",
          "url": "https://github.com/histolab/histolab/commit/72a598eb0e35536581ed1227e0ebacc57a17e915"
        },
        "date": 1607103210839,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.9806882471455785,
            "unit": "iter/sec",
            "range": "stddev: 0.00259950648814608",
            "extra": "mean: 143.25235057000327 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 303.9696291492878,
            "unit": "iter/sec",
            "range": "stddev: 0.000009819282976032176",
            "extra": "mean: 3.289802349000047 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1049.8083344767883,
            "unit": "iter/sec",
            "range": "stddev: 0.000006985187376383355",
            "extra": "mean: 952.5548303999585 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 34551.97489061501,
            "unit": "iter/sec",
            "range": "stddev: 3.2923607793748623e-7",
            "extra": "mean: 28.941905727988342 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 37527.311998641555,
            "unit": "iter/sec",
            "range": "stddev: 2.703631095660591e-7",
            "extra": "mean: 26.64725893600371 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 31533.13221013447,
            "unit": "iter/sec",
            "range": "stddev: 3.4740788766132084e-7",
            "extra": "mean: 31.712675840004522 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "e126fce4365d321e8aa93ef7df167ff1590e1d83",
          "message": "Add missing files",
          "timestamp": "2020-12-04T19:13:19+01:00",
          "tree_id": "1321bbda649f215bd5f26a08f9669099d896c3d3",
          "url": "https://github.com/histolab/histolab/commit/e126fce4365d321e8aa93ef7df167ff1590e1d83"
        },
        "date": 1607106008198,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.546491204632214,
            "unit": "iter/sec",
            "range": "stddev: 0.004581732201780963",
            "extra": "mean: 132.51191486000494 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 301.7424242476951,
            "unit": "iter/sec",
            "range": "stddev: 0.00016132315987677884",
            "extra": "mean: 3.3140848605999054 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1084.3790134079984,
            "unit": "iter/sec",
            "range": "stddev: 0.00000683507016356768",
            "extra": "mean: 922.1867886000382 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 34997.2164082084,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010991735956114843",
            "extra": "mean: 28.573701071993128 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 39671.32954715076,
            "unit": "iter/sec",
            "range": "stddev: 9.393214605615038e-7",
            "extra": "mean: 25.207120895997832 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 31537.224682482563,
            "unit": "iter/sec",
            "range": "stddev: 7.879902270079842e-7",
            "extra": "mean: 31.70856059999005 usec\nrounds: 250"
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
          "id": "861f96db81adbc4c2de2efb92897abe3104c8b23",
          "message": "fix flake8 fail on windows",
          "timestamp": "2020-12-04T23:15:05+01:00",
          "tree_id": "6ba30b66465cfa85b07ea8add85ad68cf0561825",
          "url": "https://github.com/histolab/histolab/commit/861f96db81adbc4c2de2efb92897abe3104c8b23"
        },
        "date": 1607120616376,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.804507459114488,
            "unit": "iter/sec",
            "range": "stddev: 0.003907973191430587",
            "extra": "mean: 146.9614084500006 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 260.51171006293146,
            "unit": "iter/sec",
            "range": "stddev: 0.000051337992190312474",
            "extra": "mean: 3.8385990393999236 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1033.8495841450715,
            "unit": "iter/sec",
            "range": "stddev: 0.0000061061343430371556",
            "extra": "mean: 967.2586954000053 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 34228.07969789617,
            "unit": "iter/sec",
            "range": "stddev: 1.5951691039485317e-7",
            "extra": "mean: 29.215778648004743 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 38134.892153646615,
            "unit": "iter/sec",
            "range": "stddev: 1.2923895352575176e-7",
            "extra": "mean: 26.222704288004024 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30812.140298106166,
            "unit": "iter/sec",
            "range": "stddev: 1.5066075784246688e-7",
            "extra": "mean: 32.45473992799725 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "0e43cd44499688b11680e9e56c2fa42601f959f6",
          "message": "tests: refactor test_slide and back to 100% coverage",
          "timestamp": "2020-12-08T17:46:50+01:00",
          "tree_id": "9ed107946289738e885dbe720a0f9a18e40377d9",
          "url": "https://github.com/histolab/histolab/commit/0e43cd44499688b11680e9e56c2fa42601f959f6"
        },
        "date": 1607446500327,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.941977993390595,
            "unit": "iter/sec",
            "range": "stddev: 0.007330459039930145",
            "extra": "mean: 125.9132171900012 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 265.93899891721514,
            "unit": "iter/sec",
            "range": "stddev: 0.00022528008704992495",
            "extra": "mean: 3.760260827000002 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 978.4591475047195,
            "unit": "iter/sec",
            "range": "stddev: 0.00003779237170099736",
            "extra": "mean: 1.0220150760000706 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 23545.57254598782,
            "unit": "iter/sec",
            "range": "stddev: 0.000002633355715610607",
            "extra": "mean: 42.47082962399236 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 25616.475593765037,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030351424200573984",
            "extra": "mean: 39.03737640799409 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 23230.13448979981,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034669588195213757",
            "extra": "mean: 43.04753381600494 usec\nrounds: 250"
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
          "id": "b5a940e7bfee5e5646ae68196992c7d43c41fb09",
          "message": "Add Slack user group link",
          "timestamp": "2020-12-09T14:42:08+01:00",
          "tree_id": "b0b0046d0673f7851fce9c1d895ce981bc165cd4",
          "url": "https://github.com/histolab/histolab/commit/b5a940e7bfee5e5646ae68196992c7d43c41fb09"
        },
        "date": 1607521772440,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.0140044261563155,
            "unit": "iter/sec",
            "range": "stddev: 0.001927020964530717",
            "extra": "mean: 142.57190888999787 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 291.71929940412144,
            "unit": "iter/sec",
            "range": "stddev: 0.00006857497765739945",
            "extra": "mean: 3.427952836999964 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1045.28337108499,
            "unit": "iter/sec",
            "range": "stddev: 0.000043033192080286856",
            "extra": "mean: 956.6783780000378 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 34709.79219247662,
            "unit": "iter/sec",
            "range": "stddev: 4.378825577938538e-7",
            "extra": "mean: 28.810313656004837 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 38839.62079022069,
            "unit": "iter/sec",
            "range": "stddev: 3.720127687965168e-7",
            "extra": "mean: 25.746904311995422 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 31336.3821609174,
            "unit": "iter/sec",
            "range": "stddev: 4.4142535383550735e-7",
            "extra": "mean: 31.91178850400911 usec\nrounds: 250"
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
          "id": "66267a959712a32959358da6a4882a7ca5a894b6",
          "message": "release: prepare v0.2.0 release",
          "timestamp": "2020-12-09T16:41:24+01:00",
          "tree_id": "9ec564fbb6c8af30af3c2e6d87345a708f63df7a",
          "url": "https://github.com/histolab/histolab/commit/66267a959712a32959358da6a4882a7ca5a894b6"
        },
        "date": 1607528959475,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.114503089144241,
            "unit": "iter/sec",
            "range": "stddev: 0.0023312893950327845",
            "extra": "mean: 140.55795429 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 288.33897411954234,
            "unit": "iter/sec",
            "range": "stddev: 0.00006325114110742104",
            "extra": "mean: 3.4681402438000304 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1046.7822093256425,
            "unit": "iter/sec",
            "range": "stddev: 0.000009123672523066118",
            "extra": "mean: 955.3085552000539 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 36176.56058606929,
            "unit": "iter/sec",
            "range": "stddev: 6.182791335499881e-7",
            "extra": "mean: 27.64220765600021 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 39505.090453876226,
            "unit": "iter/sec",
            "range": "stddev: 5.665948676234641e-7",
            "extra": "mean: 25.3131935279971 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 33138.76152520792,
            "unit": "iter/sec",
            "range": "stddev: 9.211148127666934e-7",
            "extra": "mean: 30.176142800005437 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "b7c78c711bb67a1130cd3cbbaf22dc7d37d2d01a",
          "message": "address Alessia's comment",
          "timestamp": "2020-12-16T15:11:42+01:00",
          "tree_id": "c914eb05960c5e4f529c9b1ad29f611afe0cab57",
          "url": "https://github.com/histolab/histolab/commit/b7c78c711bb67a1130cd3cbbaf22dc7d37d2d01a"
        },
        "date": 1608128297599,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.9161933126432835,
            "unit": "iter/sec",
            "range": "stddev: 0.0010490169691161973",
            "extra": "mean: 126.32334260999642 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 329.8087292499922,
            "unit": "iter/sec",
            "range": "stddev: 0.000054485540195182",
            "extra": "mean: 3.032060437800021 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1201.7184840622226,
            "unit": "iter/sec",
            "range": "stddev: 0.000008640117023087074",
            "extra": "mean: 832.1416482000473 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 39454.256066849244,
            "unit": "iter/sec",
            "range": "stddev: 1.9408160925857366e-7",
            "extra": "mean: 25.345808023997506 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 43758.85655408804,
            "unit": "iter/sec",
            "range": "stddev: 8.558017530315463e-8",
            "extra": "mean: 22.8525166959962 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 35210.29890584372,
            "unit": "iter/sec",
            "range": "stddev: 1.2328575612757764e-7",
            "extra": "mean: 28.40078133599809 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "567c7dd5573889a9c3f9bf572e436a904c143097",
          "message": "fix conftest HTMLRenderer when items is empty",
          "timestamp": "2020-12-16T15:20:33+01:00",
          "tree_id": "6481606ffa3d62b284f2527f16da9cffefde7ccf",
          "url": "https://github.com/histolab/histolab/commit/567c7dd5573889a9c3f9bf572e436a904c143097"
        },
        "date": 1608128890648,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.017569219420709,
            "unit": "iter/sec",
            "range": "stddev: 0.002732045689837983",
            "extra": "mean: 142.49948503999917 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 305.10386999596284,
            "unit": "iter/sec",
            "range": "stddev: 0.000011318763254485404",
            "extra": "mean: 3.27757232320007 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1051.2327897857408,
            "unit": "iter/sec",
            "range": "stddev: 0.000008903090487506374",
            "extra": "mean: 951.2640870000041 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 35385.64652670363,
            "unit": "iter/sec",
            "range": "stddev: 6.016447354196494e-7",
            "extra": "mean: 28.260046040005363 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 38256.042011977836,
            "unit": "iter/sec",
            "range": "stddev: 2.4322872064300987e-7",
            "extra": "mean: 26.13966180000807 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 31197.60155355379,
            "unit": "iter/sec",
            "range": "stddev: 5.907343791957621e-7",
            "extra": "mean: 32.05374612799642 usec\nrounds: 250"
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
          "id": "cc41b94d5f630108d78f96b21835ef830f25a2a3",
          "message": "release: prepare v0.2.1 release",
          "timestamp": "2020-12-16T15:25:01+01:00",
          "tree_id": "982c5d24d75bf81dc50239c4fef0ec63a031b1bc",
          "url": "https://github.com/histolab/histolab/commit/cc41b94d5f630108d78f96b21835ef830f25a2a3"
        },
        "date": 1608129180737,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.831441572992273,
            "unit": "iter/sec",
            "range": "stddev: 0.005248330924341682",
            "extra": "mean: 146.3819882399997 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 258.7442730175939,
            "unit": "iter/sec",
            "range": "stddev: 0.00006654100429863995",
            "extra": "mean: 3.864819840599921 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 1021.3401394775954,
            "unit": "iter/sec",
            "range": "stddev: 0.00000614237807518707",
            "extra": "mean: 979.1057467999735 usec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 34391.712467817386,
            "unit": "iter/sec",
            "range": "stddev: 1.5755095650437112e-7",
            "extra": "mean: 29.07677252000076 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 38096.72512435034,
            "unit": "iter/sec",
            "range": "stddev: 1.3701607784361049e-7",
            "extra": "mean: 26.24897538399773 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30397.592090492875,
            "unit": "iter/sec",
            "range": "stddev: 1.6530345782868454e-7",
            "extra": "mean: 32.897342559997014 usec\nrounds: 250"
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
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "1c7a85a79eabbf8e7872c950611f748dc36c842d",
          "message": "fix np_to_pil in case float input but in a correct range.",
          "timestamp": "2020-12-30T16:34:13+01:00",
          "tree_id": "a4bc6e4da6ea842f4efec70ab54ad3c1a7d134b9",
          "url": "https://github.com/histolab/histolab/commit/1c7a85a79eabbf8e7872c950611f748dc36c842d"
        },
        "date": 1609342953052,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.995627900107849,
            "unit": "iter/sec",
            "range": "stddev: 0.0016586847818635213",
            "extra": "mean: 142.94642514999737 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 293.62627278373236,
            "unit": "iter/sec",
            "range": "stddev: 0.0001532289428319909",
            "extra": "mean: 3.4056897923999485 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 595.1334166546465,
            "unit": "iter/sec",
            "range": "stddev: 0.000007730368985910398",
            "extra": "mean: 1.6802954967999995 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 25321.9210668254,
            "unit": "iter/sec",
            "range": "stddev: 2.8717036039333333e-7",
            "extra": "mean: 39.49147449598968 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 26459.783851365133,
            "unit": "iter/sec",
            "range": "stddev: 3.885566662731434e-7",
            "extra": "mean: 37.7932036639977 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 23448.58237598316,
            "unit": "iter/sec",
            "range": "stddev: 3.9130281668320297e-7",
            "extra": "mean: 42.64650135200645 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicole.bussolaceradini@gmail.com",
            "name": "Nicole Bussola",
            "username": "nicolebussola"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "b2301466f20fc226b15d0a2663ba7f3bc2fe502b",
          "message": "fix issue #117",
          "timestamp": "2020-12-30T19:30:57+01:00",
          "tree_id": "fedecf35d838f79104cb9d59651d6c8d1bfa1042",
          "url": "https://github.com/histolab/histolab/commit/b2301466f20fc226b15d0a2663ba7f3bc2fe502b"
        },
        "date": 1609353466378,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.046712070595408,
            "unit": "iter/sec",
            "range": "stddev: 0.0012281895678810282",
            "extra": "mean: 124.27436090999663 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 265.6546339869491,
            "unit": "iter/sec",
            "range": "stddev: 0.00011201569687108423",
            "extra": "mean: 3.7642859263999413 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 601.6643624901026,
            "unit": "iter/sec",
            "range": "stddev: 0.00000994552853152849",
            "extra": "mean: 1.6620562266000092 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29284.35609334245,
            "unit": "iter/sec",
            "range": "stddev: 1.2848923917496368e-7",
            "extra": "mean: 34.14792515200088 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 30651.855041634317,
            "unit": "iter/sec",
            "range": "stddev: 1.6342945868158774e-7",
            "extra": "mean: 32.62445286400134 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26919.6219958421,
            "unit": "iter/sec",
            "range": "stddev: 1.1782790714889465e-7",
            "extra": "mean: 37.14762414399638 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "7c64c363836574f28e7fec5ac13097fc53f68825",
          "message": "fix docstring",
          "timestamp": "2020-12-31T13:33:15+01:00",
          "tree_id": "884ec2a095bd790e0a75c49914b99d43f23030e0",
          "url": "https://github.com/histolab/histolab/commit/7c64c363836574f28e7fec5ac13097fc53f68825"
        },
        "date": 1609418530191,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.906345746881875,
            "unit": "iter/sec",
            "range": "stddev: 0.0017622072462600237",
            "extra": "mean: 144.79437268999845 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 276.252746437815,
            "unit": "iter/sec",
            "range": "stddev: 0.000149346565908819",
            "extra": "mean: 3.6198735139999827 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 558.5490610809031,
            "unit": "iter/sec",
            "range": "stddev: 0.00003707071071006263",
            "extra": "mean: 1.7903530229999889 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 25343.277928269534,
            "unit": "iter/sec",
            "range": "stddev: 1.4154013097066172e-7",
            "extra": "mean: 39.4581949040039 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 26227.44921847732,
            "unit": "iter/sec",
            "range": "stddev: 1.8583740842772606e-7",
            "extra": "mean: 38.12799299199469 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 23267.531542976998,
            "unit": "iter/sec",
            "range": "stddev: 4.032232878541003e-7",
            "extra": "mean: 42.978345088000424 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "9301e74d2a8dded2adfcd2fc42d03f218fb75b3a",
          "message": "Update scipy requirement from <=1.5.4,>=1.5.0 to >=1.5.0,<1.6.1\n\nUpdates the requirements on [scipy](https://github.com/scipy/scipy) to permit the latest version.\n- [Release notes](https://github.com/scipy/scipy/releases)\n- [Commits](https://github.com/scipy/scipy/compare/v1.5.0...v1.6.0)\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-01-01T17:32:34+01:00",
          "tree_id": "2fd4a48cd6363def4d11bd584f7e0cc5d96e6b60",
          "url": "https://github.com/histolab/histolab/commit/9301e74d2a8dded2adfcd2fc42d03f218fb75b3a"
        },
        "date": 1609519206375,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.3999392390841745,
            "unit": "iter/sec",
            "range": "stddev: 0.002573749075616181",
            "extra": "mean: 135.1362447299988 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 284.7166799712255,
            "unit": "iter/sec",
            "range": "stddev: 0.00004296680935457867",
            "extra": "mean: 3.5122634897999774 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 613.8813805882356,
            "unit": "iter/sec",
            "range": "stddev: 0.000019933720276225687",
            "extra": "mean: 1.628979199600053 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 27388.13185914793,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012157907106357654",
            "extra": "mean: 36.51216538399967 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 28756.689610326866,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013714712494701589",
            "extra": "mean: 34.774517288001334 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 25248.79931526635,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016213089305489024",
            "extra": "mean: 39.605843727997126 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "426ca069f87de3d82269ce10c2f666991789720c",
          "message": "Update numpy requirement from <=1.19.4,>=1.18.4 to >=1.18.4,<1.19.6\n\nUpdates the requirements on [numpy](https://github.com/numpy/numpy) to permit the latest version.\n- [Release notes](https://github.com/numpy/numpy/releases)\n- [Changelog](https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt)\n- [Commits](https://github.com/numpy/numpy/compare/v1.18.4...v1.19.5)\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-01-07T14:18:25+01:00",
          "tree_id": "f1518362571210aae2a75c1965fa08331691cc2b",
          "url": "https://github.com/histolab/histolab/commit/426ca069f87de3d82269ce10c2f666991789720c"
        },
        "date": 1610025867817,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.268864886895997,
            "unit": "iter/sec",
            "range": "stddev: 0.0038404736086150534",
            "extra": "mean: 107.88807607000138 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 337.35408568709255,
            "unit": "iter/sec",
            "range": "stddev: 0.00006159438601706007",
            "extra": "mean: 2.964244520600039 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 728.6781518769941,
            "unit": "iter/sec",
            "range": "stddev: 0.000026662941855845275",
            "extra": "mean: 1.372347993999972 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 34060.72406541808,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018320776584024311",
            "extra": "mean: 29.359328887999254 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 35668.78915461212,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013582135924979134",
            "extra": "mean: 28.035714799998914 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30501.956141573646,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017839382676713337",
            "extra": "mean: 32.784782567994625 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "51d1465d3c1408c9857a54aaf216076e2a770bdb",
          "message": "fix typo",
          "timestamp": "2021-01-07T15:35:14+01:00",
          "tree_id": "7d3bf6b14704a7404eb930565496ae6b2624aed4",
          "url": "https://github.com/histolab/histolab/commit/51d1465d3c1408c9857a54aaf216076e2a770bdb"
        },
        "date": 1610030594350,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.77664467405662,
            "unit": "iter/sec",
            "range": "stddev: 0.0015352339111607575",
            "extra": "mean: 147.5656535200011 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 241.98755807128452,
            "unit": "iter/sec",
            "range": "stddev: 0.00004243061635906331",
            "extra": "mean: 4.132443866000005 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 574.2976222865378,
            "unit": "iter/sec",
            "range": "stddev: 0.000007014386610003978",
            "extra": "mean: 1.741257426799973 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 24324.706129121336,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012167109567048996",
            "extra": "mean: 41.11046582399649 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 25514.404434720855,
            "unit": "iter/sec",
            "range": "stddev: 1.5634963805206026e-7",
            "extra": "mean: 39.19354663200238 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 22302.070585232257,
            "unit": "iter/sec",
            "range": "stddev: 1.2005989360687784e-7",
            "extra": "mean: 44.838885976002985 usec\nrounds: 250"
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
          "id": "27acf35817e82ff06ee7e3eba42d384fed25af43",
          "message": "Change method order",
          "timestamp": "2021-01-07T16:30:08+01:00",
          "tree_id": "0376dff740dde63d73cb71fd28bf318bab90bde1",
          "url": "https://github.com/histolab/histolab/commit/27acf35817e82ff06ee7e3eba42d384fed25af43"
        },
        "date": 1610033788447,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 10.940108140584417,
            "unit": "iter/sec",
            "range": "stddev: 0.0034151443381199545",
            "extra": "mean: 91.40677469999673 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 462.36383917809,
            "unit": "iter/sec",
            "range": "stddev: 0.00010412859413167618",
            "extra": "mean: 2.162798894000071 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 611.3761373341167,
            "unit": "iter/sec",
            "range": "stddev: 0.000027505785234231307",
            "extra": "mean: 1.6356542869999202 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 18786.02337300193,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018876573732354682",
            "extra": "mean: 53.23106333600845 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 20181.84962162003,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021544280884181885",
            "extra": "mean: 49.54947235999316 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 19861.676446456884,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033081446384616877",
            "extra": "mean: 50.34821721599383 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Ernesto Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "b224ca57170d168648e58188fb215cd62b2bd64e",
          "message": "skip flake8 on expectation file",
          "timestamp": "2021-01-08T00:45:56+01:00",
          "tree_id": "1b2e9241b2101344a8421e524f605f7b2dfb51f9",
          "url": "https://github.com/histolab/histolab/commit/b224ca57170d168648e58188fb215cd62b2bd64e"
        },
        "date": 1610063569835,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.30777404841362,
            "unit": "iter/sec",
            "range": "stddev: 0.006196322211425399",
            "extra": "mean: 107.43707301000029 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 351.829252222323,
            "unit": "iter/sec",
            "range": "stddev: 0.00028770787955720854",
            "extra": "mean: 2.8422878248000076 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 553.9578203772321,
            "unit": "iter/sec",
            "range": "stddev: 0.00007074770509096775",
            "extra": "mean: 1.8051915925999993 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17690.27290606128,
            "unit": "iter/sec",
            "range": "stddev: 0.0000039136370117339725",
            "extra": "mean: 56.528240423999705 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18766.41399520821,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037220993399066464",
            "extra": "mean: 53.286685472000045 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 19602.241837391164,
            "unit": "iter/sec",
            "range": "stddev: 0.00000436470144474589",
            "extra": "mean: 51.01457314400159 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Ernesto Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "a6109ea54a8093b03bde96b2275e547a75e3988e",
          "message": "refactor docstring for _remap_level",
          "timestamp": "2021-01-08T13:37:30+01:00",
          "tree_id": "5a221dee1688324b3cbdba84e0abd4ae10304e66",
          "url": "https://github.com/histolab/histolab/commit/a6109ea54a8093b03bde96b2275e547a75e3988e"
        },
        "date": 1610109933635,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.893506615833968,
            "unit": "iter/sec",
            "range": "stddev: 0.00514165066797784",
            "extra": "mean: 126.68640804000233 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 203.74824096151502,
            "unit": "iter/sec",
            "range": "stddev: 0.0002676247844943565",
            "extra": "mean: 4.908017832600012 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 482.1226921804952,
            "unit": "iter/sec",
            "range": "stddev: 0.00004672163447010551",
            "extra": "mean: 2.0741608229998523 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17809.322386657994,
            "unit": "iter/sec",
            "range": "stddev: 0.000003323968160471894",
            "extra": "mean: 56.15036767199854 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18341.95939616261,
            "unit": "iter/sec",
            "range": "stddev: 0.000003603939576543099",
            "extra": "mean: 54.51980229599758 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16270.543318319596,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028170000556584573",
            "extra": "mean: 61.46076258400444 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Ernesto Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "e7785e538d4eb57f6a688d7837208ee9e953acb2",
          "message": "some changes in the contributing guidelines",
          "timestamp": "2021-01-08T18:15:52+01:00",
          "tree_id": "5838c138f6ea3889728c927496eed621be6e9886",
          "url": "https://github.com/histolab/histolab/commit/e7785e538d4eb57f6a688d7837208ee9e953acb2"
        },
        "date": 1610126677648,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.59271935762618,
            "unit": "iter/sec",
            "range": "stddev: 0.0036153919433883843",
            "extra": "mean: 131.70511813999724 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 194.36708487071445,
            "unit": "iter/sec",
            "range": "stddev: 0.00011476681380897848",
            "extra": "mean: 5.144904039000028 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 463.08331214132915,
            "unit": "iter/sec",
            "range": "stddev: 0.0000419851312161195",
            "extra": "mean: 2.1594386448000704 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17954.79764724426,
            "unit": "iter/sec",
            "range": "stddev: 0.000003892246564832161",
            "extra": "mean: 55.69542022399128 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18529.315682528646,
            "unit": "iter/sec",
            "range": "stddev: 0.000004600913717738831",
            "extra": "mean: 53.96853381600613 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16646.46003488075,
            "unit": "iter/sec",
            "range": "stddev: 0.000004086033885676518",
            "extra": "mean: 60.07283217600707 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "0a5c448c65aa9160580b36e347a0cdb43cb0576c",
          "message": "Refactor util import",
          "timestamp": "2021-01-14T16:30:10+01:00",
          "tree_id": "938fdcc6089eb55f546098b262ac79ef68b1a39a",
          "url": "https://github.com/histolab/histolab/commit/0a5c448c65aa9160580b36e347a0cdb43cb0576c"
        },
        "date": 1610638632844,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.77396551617052,
            "unit": "iter/sec",
            "range": "stddev: 0.007165809189664217",
            "extra": "mean: 113.97355029000153 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 305.27475656553406,
            "unit": "iter/sec",
            "range": "stddev: 0.0003403200653332265",
            "extra": "mean: 3.2757376052000153 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 565.6030734088072,
            "unit": "iter/sec",
            "range": "stddev: 0.00008374486488149222",
            "extra": "mean: 1.768024339000044 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17156.62855880123,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032223230319205824",
            "extra": "mean: 58.28650987999663 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18625.093821825038,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028666922139596477",
            "extra": "mean: 53.69100470399735 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 18270.129610420143,
            "unit": "iter/sec",
            "range": "stddev: 0.000004311480384244585",
            "extra": "mean: 54.734149199995954 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicola.bussolaceradini@gmail.com",
            "name": "Nicole Bussola"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "f1c74aeb61c2f858fe8cbd54cdce1257970bf010",
          "message": "address comments",
          "timestamp": "2021-01-18T18:13:05+01:00",
          "tree_id": "1789ae1b72db8ea215b4e755b4d251a409721ee4",
          "url": "https://github.com/histolab/histolab/commit/f1c74aeb61c2f858fe8cbd54cdce1257970bf010"
        },
        "date": 1610990493241,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.6757819062733175,
            "unit": "iter/sec",
            "range": "stddev: 0.00428593153435693",
            "extra": "mean: 130.27988708000066 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 233.50949877586558,
            "unit": "iter/sec",
            "range": "stddev: 0.0002148548155599181",
            "extra": "mean: 4.282481035000001 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 527.3852503738595,
            "unit": "iter/sec",
            "range": "stddev: 0.00006731580251217042",
            "extra": "mean: 1.896147075199974 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15896.797767103664,
            "unit": "iter/sec",
            "range": "stddev: 0.000003145078508174664",
            "extra": "mean: 62.90575087200068 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 16578.53951188461,
            "unit": "iter/sec",
            "range": "stddev: 0.0000047282372958576075",
            "extra": "mean: 60.318944215992815 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15316.972461490766,
            "unit": "iter/sec",
            "range": "stddev: 0.000005419110263241667",
            "extra": "mean: 65.28705346400238 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Ernesto Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "30e8797b937a9f053e68e8bf3a1848f918c786fb",
          "message": "address CR comments",
          "timestamp": "2021-01-18T23:38:10+01:00",
          "tree_id": "9cdfae7d38bbd5d05f931603c2dce4438c1af273",
          "url": "https://github.com/histolab/histolab/commit/30e8797b937a9f053e68e8bf3a1848f918c786fb"
        },
        "date": 1611009986454,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.697829855210578,
            "unit": "iter/sec",
            "range": "stddev: 0.004627260134629331",
            "extra": "mean: 149.3020906200013 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 227.28525931702177,
            "unit": "iter/sec",
            "range": "stddev: 0.00009169199601018718",
            "extra": "mean: 4.3997573930000495 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 515.5823628466691,
            "unit": "iter/sec",
            "range": "stddev: 0.000072182986755854",
            "extra": "mean: 1.9395543216000077 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 23775.229141941145,
            "unit": "iter/sec",
            "range": "stddev: 0.000004305754651522072",
            "extra": "mean: 42.060583055997995 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 24715.230305753965,
            "unit": "iter/sec",
            "range": "stddev: 0.000003192183147516055",
            "extra": "mean: 40.46088131200577 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 21913.266497417702,
            "unit": "iter/sec",
            "range": "stddev: 0.000004313315624000368",
            "extra": "mean: 45.63445619199865 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicola.bussolaceradini@gmail.com",
            "name": "Nicole Bussola"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "ddd0c4a7116367d24f81f38755cd6483913bbf7e",
          "message": "fix quickstart bug",
          "timestamp": "2021-01-19T16:10:10+01:00",
          "tree_id": "c3f3b360b88c725c0b3ad264f0e98f00c5429c35",
          "url": "https://github.com/histolab/histolab/commit/ddd0c4a7116367d24f81f38755cd6483913bbf7e"
        },
        "date": 1611069427807,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.9879071329527465,
            "unit": "iter/sec",
            "range": "stddev: 0.001567984530589608",
            "extra": "mean: 125.18923709999969 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 210.6323822753285,
            "unit": "iter/sec",
            "range": "stddev: 0.0000887556312845274",
            "extra": "mean: 4.747608080000009 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 664.597152074359,
            "unit": "iter/sec",
            "range": "stddev: 0.00001822185419435431",
            "extra": "mean: 1.5046709075998477 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29328.093222382864,
            "unit": "iter/sec",
            "range": "stddev: 1.244235882128132e-7",
            "extra": "mean: 34.097000184001445 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 30621.05480290236,
            "unit": "iter/sec",
            "range": "stddev: 1.0354656894044006e-7",
            "extra": "mean: 32.657268223993924 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26857.545318227876,
            "unit": "iter/sec",
            "range": "stddev: 1.2437475118373388e-7",
            "extra": "mean: 37.23348459999852 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Ernesto Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Ernesto Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "e79f57df869eefedd769b7932d87d9e5ce84e947",
          "message": "release: prepare v0.2.2 release",
          "timestamp": "2021-01-19T22:44:47+01:00",
          "tree_id": "df90e5fa7312c1ebe069137c58c23f21cae1ae88",
          "url": "https://github.com/histolab/histolab/commit/e79f57df869eefedd769b7932d87d9e5ce84e947"
        },
        "date": 1611093146032,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.680010549213166,
            "unit": "iter/sec",
            "range": "stddev: 0.004396137253671233",
            "extra": "mean: 130.20815448000292 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 296.3859125068011,
            "unit": "iter/sec",
            "range": "stddev: 0.00004236560138210492",
            "extra": "mean: 3.373979523999992 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 629.6919192627149,
            "unit": "iter/sec",
            "range": "stddev: 0.000032852240475497943",
            "extra": "mean: 1.588078184599965 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 28625.283689446875,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017792644915394978",
            "extra": "mean: 34.93415159999495 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29592.95901686731,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017710259379632353",
            "extra": "mean: 33.79182187999595 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 25902.263533574147,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019713387031530804",
            "extra": "mean: 38.60666457600564 usec\nrounds: 250"
          }
        ]
      }
    ]
  }
}