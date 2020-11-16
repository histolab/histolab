window.BENCHMARK_DATA = {
  "lastUpdate": 1605524080273,
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
      }
    ]
  }
}