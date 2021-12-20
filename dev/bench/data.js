window.BENCHMARK_DATA = {
  "lastUpdate": 1640002618615,
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
          "id": "d0fa9038905c1d3da5e073f645070dec30549045",
          "message": "Update README.md",
          "timestamp": "2021-01-19T22:51:47+01:00",
          "tree_id": "2c510db064ef56121463bfb68614926d82909e47",
          "url": "https://github.com/histolab/histolab/commit/d0fa9038905c1d3da5e073f645070dec30549045"
        },
        "date": 1611093625242,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.657046452101279,
            "unit": "iter/sec",
            "range": "stddev: 0.00783728438579432",
            "extra": "mean: 130.59865919000345 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 242.85895078520556,
            "unit": "iter/sec",
            "range": "stddev: 0.00018481691720386513",
            "extra": "mean: 4.117616405600142 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 529.2640091758047,
            "unit": "iter/sec",
            "range": "stddev: 0.00005103050555402954",
            "extra": "mean: 1.889416213200002 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15550.048933249453,
            "unit": "iter/sec",
            "range": "stddev: 0.000004904938823525184",
            "extra": "mean: 64.30847930399614 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 16795.59586368971,
            "unit": "iter/sec",
            "range": "stddev: 0.000005080671409356078",
            "extra": "mean: 59.53941784000017 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15605.274176586034,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043595214980911105",
            "extra": "mean: 64.08089910399576 usec\nrounds: 250"
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
          "id": "01ce4ef4d0f27a57132c1f8ffacbb571fa398ef2",
          "message": "Update numpy requirement from <1.19.6,>=1.18.4 to >=1.18.4,<1.20.1\n\nUpdates the requirements on [numpy](https://github.com/numpy/numpy) to permit the latest version.\n- [Release notes](https://github.com/numpy/numpy/releases)\n- [Changelog](https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt)\n- [Commits](https://github.com/numpy/numpy/compare/v1.18.4...v1.20.0)\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-02-01T09:54:02+01:00",
          "tree_id": "f9781e171c8e149887963a721d61ce2c02193b21",
          "url": "https://github.com/histolab/histolab/commit/01ce4ef4d0f27a57132c1f8ffacbb571fa398ef2"
        },
        "date": 1612170148487,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.455683492622269,
            "unit": "iter/sec",
            "range": "stddev: 0.005551091584103586",
            "extra": "mean: 134.12586531999978 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 213.56738443375045,
            "unit": "iter/sec",
            "range": "stddev: 0.00020565311515134178",
            "extra": "mean: 4.6823629115999434 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 462.38027929821345,
            "unit": "iter/sec",
            "range": "stddev: 0.00008139314102026439",
            "extra": "mean: 2.1627219947999716 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 18096.292366619396,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036875732683015756",
            "extra": "mean: 55.259938319995854 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 19261.05833547788,
            "unit": "iter/sec",
            "range": "stddev: 0.000005366722034170728",
            "extra": "mean: 51.91822705598952 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16307.743250673358,
            "unit": "iter/sec",
            "range": "stddev: 0.00000620877970327011",
            "extra": "mean: 61.32056316000126 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "6febdd858c933157e8cf5d6dea72982fcbecf6d5",
          "message": "Update numpy requirement from <1.20.1,>=1.18.4 to >=1.18.4,<1.20.2\n\nUpdates the requirements on [numpy](https://github.com/numpy/numpy) to permit the latest version.\n- [Release notes](https://github.com/numpy/numpy/releases)\n- [Changelog](https://github.com/numpy/numpy/blob/master/doc/HOWTO_RELEASE.rst.txt)\n- [Commits](https://github.com/numpy/numpy/compare/v1.18.4...v1.20.1)\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-02-08T09:48:28+01:00",
          "tree_id": "6a2b9bef40ba517e6896fcb1ccabbf490c9158e2",
          "url": "https://github.com/histolab/histolab/commit/6febdd858c933157e8cf5d6dea72982fcbecf6d5"
        },
        "date": 1612774570352,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.439235492833696,
            "unit": "iter/sec",
            "range": "stddev: 0.006502614261713384",
            "extra": "mean: 118.4941456899935 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 326.56199450600934,
            "unit": "iter/sec",
            "range": "stddev: 0.0001905144204735854",
            "extra": "mean: 3.062205697000047 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 550.0685448719619,
            "unit": "iter/sec",
            "range": "stddev: 0.00005962080538557291",
            "extra": "mean: 1.8179552518000603 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 18102.36917952937,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029816836482597606",
            "extra": "mean: 55.24138802399557 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 19146.468206103218,
            "unit": "iter/sec",
            "range": "stddev: 0.000002790667659426323",
            "extra": "mean: 52.228953624002315 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 18011.522648847353,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033286570996708623",
            "extra": "mean: 55.52001457600227 usec\nrounds: 250"
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
          "id": "4f59b9a02c47b4032bfb9b6158326d8d7106d2cf",
          "message": "Address CR comments",
          "timestamp": "2021-02-16T09:21:50+01:00",
          "tree_id": "a07825d86884e917546e0be1bd620f4942aa5a22",
          "url": "https://github.com/histolab/histolab/commit/4f59b9a02c47b4032bfb9b6158326d8d7106d2cf"
        },
        "date": 1613464132043,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.481312873566992,
            "unit": "iter/sec",
            "range": "stddev: 0.005703730345110216",
            "extra": "mean: 105.47062556999947 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 463.58415392500893,
            "unit": "iter/sec",
            "range": "stddev: 0.00013074715443224796",
            "extra": "mean: 2.1571056549999414 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 582.7870411063856,
            "unit": "iter/sec",
            "range": "stddev: 0.00005654807306713238",
            "extra": "mean: 1.7158926493999616 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 19413.5982407535,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022417146273924688",
            "extra": "mean: 51.510286120003 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 19826.17077403888,
            "unit": "iter/sec",
            "range": "stddev: 0.000003162820024633342",
            "extra": "mean: 50.43838325600609 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 20442.993430061088,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029881901066576914",
            "extra": "mean: 48.91651525600537 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "7036066adfaede91abc9ce277088f0f5dbf105da",
          "message": "address CR comments part 2",
          "timestamp": "2021-02-16T15:37:13+01:00",
          "tree_id": "e77ab603ec26326dd9a1df7e1711f99d7b90c841",
          "url": "https://github.com/histolab/histolab/commit/7036066adfaede91abc9ce277088f0f5dbf105da"
        },
        "date": 1613486773368,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.364684799963309,
            "unit": "iter/sec",
            "range": "stddev: 0.005311244329859465",
            "extra": "mean: 135.78313630000594 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 246.81349172064213,
            "unit": "iter/sec",
            "range": "stddev: 0.00047283082352206904",
            "extra": "mean: 4.0516423677999684 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 510.7145600132294,
            "unit": "iter/sec",
            "range": "stddev: 0.00008483184947657829",
            "extra": "mean: 1.9580409064000377 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17282.57547321791,
            "unit": "iter/sec",
            "range": "stddev: 0.000004606030683978216",
            "extra": "mean: 57.861746448013946 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18244.48554024723,
            "unit": "iter/sec",
            "range": "stddev: 0.00000391107817938169",
            "extra": "mean: 54.81108238398974 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16791.729087199867,
            "unit": "iter/sec",
            "range": "stddev: 0.000006170076942022662",
            "extra": "mean: 59.55312849599795 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "43278e75af58dc63ac663c06b2d3e2b19cfcbab5",
          "message": "release: prepare v0.2.3 release",
          "timestamp": "2021-02-16T17:36:57+01:00",
          "tree_id": "7462a252af1af46df7298878cfd6b51522afcba4",
          "url": "https://github.com/histolab/histolab/commit/43278e75af58dc63ac663c06b2d3e2b19cfcbab5"
        },
        "date": 1613493884477,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.572527697070866,
            "unit": "iter/sec",
            "range": "stddev: 0.005168655488912557",
            "extra": "mean: 132.05630141000483 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 280.15439154321257,
            "unit": "iter/sec",
            "range": "stddev: 0.00007979294678990037",
            "extra": "mean: 3.5694603767999635 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 633.1023694177519,
            "unit": "iter/sec",
            "range": "stddev: 0.000044653350698737805",
            "extra": "mean: 1.5795233887999416 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 28901.06883819324,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020970228134205735",
            "extra": "mean: 34.60079644800135 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29663.76161333509,
            "unit": "iter/sec",
            "range": "stddev: 0.00000139367010717262",
            "extra": "mean: 33.71116627199626 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 25858.784175886365,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019032317444961843",
            "extra": "mean: 38.67157841599192 usec\nrounds: 250"
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
          "id": "b6e52ec75baf3c2d77eaf724ef64f43776b6c55d",
          "message": "Update scipy requirement from <1.6.1,>=1.5.0 to >=1.5.0,<1.6.2\n\nUpdates the requirements on [scipy](https://github.com/scipy/scipy) to permit the latest version.\n- [Release notes](https://github.com/scipy/scipy/releases)\n- [Commits](https://github.com/scipy/scipy/compare/v1.5.0...v1.6.1)\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-02-18T09:57:42+01:00",
          "tree_id": "71eb9aa4ad315482f8d0823ab7ff1cc962890cae",
          "url": "https://github.com/histolab/histolab/commit/b6e52ec75baf3c2d77eaf724ef64f43776b6c55d"
        },
        "date": 1613639201340,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.810361779832125,
            "unit": "iter/sec",
            "range": "stddev: 0.006333286106577653",
            "extra": "mean: 128.03504219000388 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 317.43465789679476,
            "unit": "iter/sec",
            "range": "stddev: 0.0002802391638539448",
            "extra": "mean: 3.1502546275999976 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 532.900021383705,
            "unit": "iter/sec",
            "range": "stddev: 0.00003846565511153296",
            "extra": "mean: 1.8765246010000967 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17046.061761039997,
            "unit": "iter/sec",
            "range": "stddev: 0.00000381840343232142",
            "extra": "mean: 58.664576840004884 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18012.20308170241,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033581194746610764",
            "extra": "mean: 55.517917239998496 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 17206.50395063001,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035736374914963614",
            "extra": "mean: 58.117558503997316 usec\nrounds: 250"
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
          "id": "2a4706d639e6a6e5f95db26cce16c6ed7e401686",
          "message": "Fix typo in workflow name",
          "timestamp": "2021-02-18T20:58:10+01:00",
          "tree_id": "a705c7501dba3339e281c50af0a4cd0fc11a876e",
          "url": "https://github.com/histolab/histolab/commit/2a4706d639e6a6e5f95db26cce16c6ed7e401686"
        },
        "date": 1613678860428,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.474365507252621,
            "unit": "iter/sec",
            "range": "stddev: 0.004315765558514488",
            "extra": "mean: 118.00293475000217 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 223.62092981305705,
            "unit": "iter/sec",
            "range": "stddev: 0.0001707653568460096",
            "extra": "mean: 4.471853331599959 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 521.8808282861633,
            "unit": "iter/sec",
            "range": "stddev: 0.000030220766823473607",
            "extra": "mean: 1.9161462651999728 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 20441.210762641258,
            "unit": "iter/sec",
            "range": "stddev: 0.000002412866036620113",
            "extra": "mean: 48.920781240004544 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 21705.015751801675,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019634430337503317",
            "extra": "mean: 46.07230012800119 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 19422.075557490745,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020791616139253804",
            "extra": "mean: 51.487802992009165 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "9a67ff80c153e28c385cfa0e9d262393d40ca047",
          "message": "fix docstring in tiler according to CR",
          "timestamp": "2021-03-09T17:32:50+01:00",
          "tree_id": "a10a689dc34727892850f28ed69196b3c7560915",
          "url": "https://github.com/histolab/histolab/commit/9a67ff80c153e28c385cfa0e9d262393d40ca047"
        },
        "date": 1615308092548,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.998768807602403,
            "unit": "iter/sec",
            "range": "stddev: 0.0012956023898364425",
            "extra": "mean: 100.01231343999734 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 325.5005366159919,
            "unit": "iter/sec",
            "range": "stddev: 0.000028278008968596698",
            "extra": "mean: 3.072191555799941 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 950.6203612824121,
            "unit": "iter/sec",
            "range": "stddev: 0.000023257263674400193",
            "extra": "mean: 1.0519446466000089 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 31757.180649701942,
            "unit": "iter/sec",
            "range": "stddev: 6.218062524608374e-7",
            "extra": "mean: 31.488941384013746 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 31757.40334023751,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020785081061065217",
            "extra": "mean: 31.48872057599783 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 28533.136925568775,
            "unit": "iter/sec",
            "range": "stddev: 3.2570544840501365e-7",
            "extra": "mean: 35.04697021601896 usec\nrounds: 250"
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
          "id": "ad2e3a3154a827346ba4c027f0bff1767458428d",
          "message": "remove leftover tests",
          "timestamp": "2021-03-10T23:18:42+01:00",
          "tree_id": "3110cf35aba744570806cc3579e986114eedbd97",
          "url": "https://github.com/histolab/histolab/commit/ad2e3a3154a827346ba4c027f0bff1767458428d"
        },
        "date": 1615415337720,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.551252701825865,
            "unit": "iter/sec",
            "range": "stddev: 0.003955460939499915",
            "extra": "mean: 116.94193060000202 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 220.62499129112942,
            "unit": "iter/sec",
            "range": "stddev: 0.0001730118636968269",
            "extra": "mean: 4.532578082600048 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 505.6589258220252,
            "unit": "iter/sec",
            "range": "stddev: 0.000049984060565318644",
            "extra": "mean: 1.9776176171998712 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 18707.135238244937,
            "unit": "iter/sec",
            "range": "stddev: 0.000003027628050246787",
            "extra": "mean: 53.45553914399443 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 19235.78824151355,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027555599375972084",
            "extra": "mean: 51.98643213600462 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 17720.317076822015,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030171912874513317",
            "extra": "mean: 56.43239879200519 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mtageld@emory.edu",
            "name": "kheffah",
            "username": "kheffah"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "f72cd7ae89c7dfe79252c27add8137a5a41dbcf7",
          "message": "fix test_tiler.py",
          "timestamp": "2021-03-15T17:48:22+01:00",
          "tree_id": "5451b823de00226756e4fbb5ca98eba5742b9b25",
          "url": "https://github.com/histolab/histolab/commit/f72cd7ae89c7dfe79252c27add8137a5a41dbcf7"
        },
        "date": 1615827534098,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.475990952244461,
            "unit": "iter/sec",
            "range": "stddev: 0.0014460521934511925",
            "extra": "mean: 133.76153160000513 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 264.0412205429252,
            "unit": "iter/sec",
            "range": "stddev: 0.00006263129236648796",
            "extra": "mean: 3.7872874468001103 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 592.3612505134488,
            "unit": "iter/sec",
            "range": "stddev: 0.000008662606279631379",
            "extra": "mean: 1.688159040000028 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26822.36615626137,
            "unit": "iter/sec",
            "range": "stddev: 1.6349330528519326e-7",
            "extra": "mean: 37.28231857600531 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 28080.422771120644,
            "unit": "iter/sec",
            "range": "stddev: 1.626778269531383e-7",
            "extra": "mean: 35.611999439996 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24609.216678833982,
            "unit": "iter/sec",
            "range": "stddev: 2.4892316730836587e-7",
            "extra": "mean: 40.63518205600121 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "47f41f3bb5a830ec14ce00418527db21bad2f91b",
          "message": "remove coverall references",
          "timestamp": "2021-03-27T09:30:57+01:00",
          "tree_id": "a1550f1ba3e002577524acb23b06ab14a4bdd861",
          "url": "https://github.com/histolab/histolab/commit/47f41f3bb5a830ec14ce00418527db21bad2f91b"
        },
        "date": 1616834490984,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.536858563650867,
            "unit": "iter/sec",
            "range": "stddev: 0.00600857718199367",
            "extra": "mean: 117.13910831999783 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 366.07207230459176,
            "unit": "iter/sec",
            "range": "stddev: 0.0002875567206988334",
            "extra": "mean: 2.7317025134000006 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 538.1043392297167,
            "unit": "iter/sec",
            "range": "stddev: 0.000034946526936557444",
            "extra": "mean: 1.8583756477999713 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16746.102579038037,
            "unit": "iter/sec",
            "range": "stddev: 0.000001913217133937925",
            "extra": "mean: 59.715387223995094 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 17618.52095682145,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020557013781980102",
            "extra": "mean: 56.75845336000384 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16883.152706620782,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030453040747679324",
            "extra": "mean: 59.2306435520095 usec\nrounds: 250"
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
          "id": "75291a74d2d509ef4dd9b6eababc5a89754655f3",
          "message": "Update scipy requirement from <1.6.2,>=1.5.0 to >=1.5.0,<1.6.3\n\nUpdates the requirements on [scipy](https://github.com/scipy/scipy) to permit the latest version.\n- [Release notes](https://github.com/scipy/scipy/releases)\n- [Commits](https://github.com/scipy/scipy/compare/v1.5.0...v1.6.2)\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-03-27T09:41:55+01:00",
          "tree_id": "1587a412e385adeb2af16490a33afac523406c38",
          "url": "https://github.com/histolab/histolab/commit/75291a74d2d509ef4dd9b6eababc5a89754655f3"
        },
        "date": 1616835123113,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.931248488956448,
            "unit": "iter/sec",
            "range": "stddev: 0.003384444191528575",
            "extra": "mean: 126.08355435999896 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 278.1961283582934,
            "unit": "iter/sec",
            "range": "stddev: 0.00010798788952569111",
            "extra": "mean: 3.5945863297999723 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 604.1568828972631,
            "unit": "iter/sec",
            "range": "stddev: 0.000027985997756820453",
            "extra": "mean: 1.6551992177999408 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 27532.54814915219,
            "unit": "iter/sec",
            "range": "stddev: 7.496508165052084e-7",
            "extra": "mean: 36.320648367985996 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29092.005275967706,
            "unit": "iter/sec",
            "range": "stddev: 6.819364798288319e-7",
            "extra": "mean: 34.37370475200896 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 25710.905075234587,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011589889572504531",
            "extra": "mean: 38.89400225600093 usec\nrounds: 250"
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
          "id": "ae86a348b41006a9e17f2f7db3ee3cb96b5995a4",
          "message": "Update README.md",
          "timestamp": "2021-03-27T16:17:04+01:00",
          "tree_id": "f9594d82b28b155ee73caf0451dd23e2488fe4e8",
          "url": "https://github.com/histolab/histolab/commit/ae86a348b41006a9e17f2f7db3ee3cb96b5995a4"
        },
        "date": 1616858855187,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.493938715954747,
            "unit": "iter/sec",
            "range": "stddev: 0.0016410451324911574",
            "extra": "mean: 133.44117665000113 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 289.4406589292756,
            "unit": "iter/sec",
            "range": "stddev: 0.00007238049313086201",
            "extra": "mean: 3.4549396194000113 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 589.3335665843318,
            "unit": "iter/sec",
            "range": "stddev: 0.000008640180296070425",
            "extra": "mean: 1.6968319075999942 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 27376.72654698691,
            "unit": "iter/sec",
            "range": "stddev: 1.528015825632584e-7",
            "extra": "mean: 36.52737657600119 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 28335.816381275745,
            "unit": "iter/sec",
            "range": "stddev: 1.9340451256905278e-7",
            "extra": "mean: 35.29102484800114 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24800.8111198495,
            "unit": "iter/sec",
            "range": "stddev: 1.6814022120240358e-7",
            "extra": "mean: 40.321261880005295 usec\nrounds: 250"
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
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "e1ff656cd5cd770189eac0b4199ad54269e86427",
          "message": "Update requirements.txt",
          "timestamp": "2021-03-31T12:29:47+02:00",
          "tree_id": "bcf8083874e3f8df379ce5f72c2b6167c27ed902",
          "url": "https://github.com/histolab/histolab/commit/e1ff656cd5cd770189eac0b4199ad54269e86427"
        },
        "date": 1617187192267,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.368459452329981,
            "unit": "iter/sec",
            "range": "stddev: 0.004114068435659094",
            "extra": "mean: 119.49630701999467 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 258.9899938201706,
            "unit": "iter/sec",
            "range": "stddev: 0.00016630841516726308",
            "extra": "mean: 3.861153032399966 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 518.7639614114921,
            "unit": "iter/sec",
            "range": "stddev: 0.00005285561806747304",
            "extra": "mean: 1.9276589631999967 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 19536.99732976539,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019000522763037053",
            "extra": "mean: 51.184938152008726 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 19598.48264370561,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019265395335703656",
            "extra": "mean: 51.02435827199952 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 18241.95366432045,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028037320388445904",
            "extra": "mean: 54.818689839998115 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "0f0c18c2b43a5b993674323f6eda2b7b50f53a23",
          "message": "fix issue 238",
          "timestamp": "2021-04-03T18:50:40+02:00",
          "tree_id": "bdcf4cda1fb1ba2e34370d9bcd053cd8dbd2480b",
          "url": "https://github.com/histolab/histolab/commit/0f0c18c2b43a5b993674323f6eda2b7b50f53a23"
        },
        "date": 1617469225088,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.564230705894946,
            "unit": "iter/sec",
            "range": "stddev: 0.003010813172701874",
            "extra": "mean: 132.20115023999483 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 310.4450291917683,
            "unit": "iter/sec",
            "range": "stddev: 0.00002906231975380221",
            "extra": "mean: 3.2211821932000704 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 624.3236907485039,
            "unit": "iter/sec",
            "range": "stddev: 0.000023160735090460387",
            "extra": "mean: 1.601733227199975 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 28354.17213196129,
            "unit": "iter/sec",
            "range": "stddev: 0.000001221704898978402",
            "extra": "mean: 35.268178359994636 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29792.064133811487,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012172990315188184",
            "extra": "mean: 33.56598574400505 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26233.500588024584,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014146434030793043",
            "extra": "mean: 38.119197879999774 usec\nrounds: 250"
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
          "id": "4e30415c6ae078b1ad694eba1c4f1e16f37ace14",
          "message": "Remove Deprecated package",
          "timestamp": "2021-04-16T23:19:42+02:00",
          "tree_id": "adfab5be80b33e7d890296d02bccb8043427e0c3",
          "url": "https://github.com/histolab/histolab/commit/4e30415c6ae078b1ad694eba1c4f1e16f37ace14"
        },
        "date": 1618608647288,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.294480175480322,
            "unit": "iter/sec",
            "range": "stddev: 0.0053544908221619305",
            "extra": "mean: 137.08996062000438 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 258.1777052592547,
            "unit": "iter/sec",
            "range": "stddev: 0.00019576336513325767",
            "extra": "mean: 3.873301139600062 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 528.3227741310367,
            "unit": "iter/sec",
            "range": "stddev: 0.00003126837476252969",
            "extra": "mean: 1.892782308399933 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16706.534588379865,
            "unit": "iter/sec",
            "range": "stddev: 0.000002676948250686397",
            "extra": "mean: 59.856817983996734 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 17498.698416365263,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026542972040461016",
            "extra": "mean: 57.14710752799601 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16863.074761179145,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026978559608071343",
            "extra": "mean: 59.30116625599749 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "50be1c123f926a00359ab7c7272b4256ead52ecb",
          "message": "release: prepare v0.2.4 release",
          "timestamp": "2021-04-19T22:00:48+02:00",
          "tree_id": "07d5771e5cb3a14499bcd5dc010a25121fbc1ec3",
          "url": "https://github.com/histolab/histolab/commit/50be1c123f926a00359ab7c7272b4256ead52ecb"
        },
        "date": 1618863077195,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.855679226040706,
            "unit": "iter/sec",
            "range": "stddev: 0.006220268852732919",
            "extra": "mean: 145.86446754999656 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 278.3127767431719,
            "unit": "iter/sec",
            "range": "stddev: 0.0000944975882992126",
            "extra": "mean: 3.593079741800011 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 600.4580572673663,
            "unit": "iter/sec",
            "range": "stddev: 0.00001803809992071927",
            "extra": "mean: 1.6653952559999197 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26517.728170476243,
            "unit": "iter/sec",
            "range": "stddev: 6.192623551824393e-7",
            "extra": "mean: 37.710621120000724 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 28027.879496758655,
            "unit": "iter/sec",
            "range": "stddev: 7.500057718864209e-7",
            "extra": "mean: 35.67876050400628 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24336.558797579863,
            "unit": "iter/sec",
            "range": "stddev: 8.147938677058876e-7",
            "extra": "mean: 41.09044373600773 usec\nrounds: 250"
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
          "id": "21730a68caa6c4ce13c5e381c1c511dc2df51a78",
          "message": "Fix random choice of coordinates within true values of a binary mask",
          "timestamp": "2021-04-24T23:13:43+02:00",
          "tree_id": "5ba4781daac5e7ab124cf93742aad5daa38dcd16",
          "url": "https://github.com/histolab/histolab/commit/21730a68caa6c4ce13c5e381c1c511dc2df51a78"
        },
        "date": 1619299488295,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.412930832139085,
            "unit": "iter/sec",
            "range": "stddev: 0.0049265754797791945",
            "extra": "mean: 134.8994105899999 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 263.518408960159,
            "unit": "iter/sec",
            "range": "stddev: 0.00018611713309903314",
            "extra": "mean: 3.794801296600076 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 514.5538174891202,
            "unit": "iter/sec",
            "range": "stddev: 0.00003925166602298671",
            "extra": "mean: 1.9434313107999515 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15249.575178708788,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027792618596480967",
            "extra": "mean: 65.57559723999293 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 15689.750322829183,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024807851631687557",
            "extra": "mean: 63.73587720799878 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15023.544103879425,
            "unit": "iter/sec",
            "range": "stddev: 0.000004455772291671479",
            "extra": "mean: 66.56219019197852 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "805aba68234b4f5d8c0eed851cb545238f46946e",
          "message": "Update scipy requirement from <1.6.3,>=1.5.0 to >=1.5.0,<1.6.4\n\nUpdates the requirements on [scipy](https://github.com/scipy/scipy) to permit the latest version.\n- [Release notes](https://github.com/scipy/scipy/releases)\n- [Commits](https://github.com/scipy/scipy/compare/v1.5.0...v1.6.3)\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-04-26T11:08:04+02:00",
          "tree_id": "6208cbf27d11f11b8f118bf2a8a3aa436b0e9d2e",
          "url": "https://github.com/histolab/histolab/commit/805aba68234b4f5d8c0eed851cb545238f46946e"
        },
        "date": 1619428701459,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.845356056965885,
            "unit": "iter/sec",
            "range": "stddev: 0.004673664119233177",
            "extra": "mean: 127.46394079999732 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 214.41409212283642,
            "unit": "iter/sec",
            "range": "stddev: 0.00033133607254005805",
            "extra": "mean: 4.663872556599995 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 500.8300161200656,
            "unit": "iter/sec",
            "range": "stddev: 0.000044865363869511844",
            "extra": "mean: 1.9966854377998517 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17967.068668661912,
            "unit": "iter/sec",
            "range": "stddev: 0.000002238287544650595",
            "extra": "mean: 55.65738176000832 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18372.347630838416,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025325081551007088",
            "extra": "mean: 54.429625439999654 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 17182.186644176312,
            "unit": "iter/sec",
            "range": "stddev: 0.000002963297389052848",
            "extra": "mean: 58.199810111999795 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "a3694106d1437bfb0bc08baa3a5ccf0af6b2db9c",
          "message": "move local otsu filter",
          "timestamp": "2021-04-26T13:10:55+02:00",
          "tree_id": "51d1e3fcf17b1fea8e7a5602bd7a4149b1bfbc1e",
          "url": "https://github.com/histolab/histolab/commit/a3694106d1437bfb0bc08baa3a5ccf0af6b2db9c"
        },
        "date": 1619435988015,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.944457486643865,
            "unit": "iter/sec",
            "range": "stddev: 0.001264202199662504",
            "extra": "mean: 125.87391923000268 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 235.2410400064009,
            "unit": "iter/sec",
            "range": "stddev: 0.00013738563086430653",
            "extra": "mean: 4.250958931200057 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 684.896989917403,
            "unit": "iter/sec",
            "range": "stddev: 0.000008555315359209592",
            "extra": "mean: 1.4600735800001075 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 30315.26744840456,
            "unit": "iter/sec",
            "range": "stddev: 8.311456463570997e-8",
            "extra": "mean: 32.986679127999196 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 31815.11917217715,
            "unit": "iter/sec",
            "range": "stddev: 9.304464947704424e-8",
            "extra": "mean: 31.431596864000312 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27792.817206136795,
            "unit": "iter/sec",
            "range": "stddev: 8.277787558487125e-8",
            "extra": "mean: 35.9805194479959 usec\nrounds: 250"
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
          "id": "b2536c81e3f389c590b0ab668edef012e906bbd1",
          "message": "fix flake8",
          "timestamp": "2021-04-26T16:59:58+02:00",
          "tree_id": "8ca9a5019ef24e8a6b227d6734c27cd5ae320712",
          "url": "https://github.com/histolab/histolab/commit/b2536c81e3f389c590b0ab668edef012e906bbd1"
        },
        "date": 1619449794559,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.410794906147077,
            "unit": "iter/sec",
            "range": "stddev: 0.0034047895662275137",
            "extra": "mean: 118.8948263699956 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 221.23722381702407,
            "unit": "iter/sec",
            "range": "stddev: 0.00020524672254489996",
            "extra": "mean: 4.520035022800039 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 512.7275889119882,
            "unit": "iter/sec",
            "range": "stddev: 0.000031869368354163573",
            "extra": "mean: 1.950353407200123 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17135.991210622316,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029052614167301667",
            "extra": "mean: 58.35670593599025 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18015.94109877918,
            "unit": "iter/sec",
            "range": "stddev: 0.000002383048508087805",
            "extra": "mean: 55.50639816799594 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16602.829718369638,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024188720526428096",
            "extra": "mean: 60.23069663200749 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "984d91ef9a5e295dd2310f5c28eb0b3620fef60a",
          "message": "address CR comments",
          "timestamp": "2021-04-26T18:22:05+02:00",
          "tree_id": "ce81f526a3a7d0f0628a4e56d744ad7e8169c5e0",
          "url": "https://github.com/histolab/histolab/commit/984d91ef9a5e295dd2310f5c28eb0b3620fef60a"
        },
        "date": 1619454731594,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.346256431969127,
            "unit": "iter/sec",
            "range": "stddev: 0.007125100773161135",
            "extra": "mean: 119.8141955199992 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 373.524251216056,
            "unit": "iter/sec",
            "range": "stddev: 0.00023103468586655146",
            "extra": "mean: 2.6772023415999686 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 541.3758796600821,
            "unit": "iter/sec",
            "range": "stddev: 0.00006285931893076066",
            "extra": "mean: 1.8471454631999449 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15465.038822716611,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038101718124321407",
            "extra": "mean: 64.66197799200472 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 15912.26759600647,
            "unit": "iter/sec",
            "range": "stddev: 0.000004366357303917348",
            "extra": "mean: 62.84459420799158 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15894.525089140627,
            "unit": "iter/sec",
            "range": "stddev: 0.00000518619696002671",
            "extra": "mean: 62.91474544799166 usec\nrounds: 250"
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
          "id": "4f759629caa60f1dece5b46fee1e6ac39729792d",
          "message": "fix random tiler coordinates",
          "timestamp": "2021-04-27T13:06:58+02:00",
          "tree_id": "a6a5966b781fb53df774567a3e9fe785ea3e2844",
          "url": "https://github.com/histolab/histolab/commit/4f759629caa60f1dece5b46fee1e6ac39729792d"
        },
        "date": 1619522188900,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.302104834227228,
            "unit": "iter/sec",
            "range": "stddev: 0.005293941513342305",
            "extra": "mean: 107.5025510699993 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 454.53533282868887,
            "unit": "iter/sec",
            "range": "stddev: 0.00022049208728952872",
            "extra": "mean: 2.2000489902000484 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 573.3947616600152,
            "unit": "iter/sec",
            "range": "stddev: 0.00006412796311680666",
            "extra": "mean: 1.7439991902000203 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17504.149502172088,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029124803404423328",
            "extra": "mean: 57.12931096000466 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 17695.288697697077,
            "unit": "iter/sec",
            "range": "stddev: 0.000002624215553309469",
            "extra": "mean: 56.51221729601639 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 17767.435362814143,
            "unit": "iter/sec",
            "range": "stddev: 0.000004559286438421917",
            "extra": "mean: 56.28274309599692 usec\nrounds: 250"
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
          "id": "b6dcc609b56dc72561b203ddf235502ef4f0a481",
          "message": "address comment",
          "timestamp": "2021-04-27T16:10:13+02:00",
          "tree_id": "d95311c337211bfff7bc9e8b7188715fa6b07abe",
          "url": "https://github.com/histolab/histolab/commit/b6dcc609b56dc72561b203ddf235502ef4f0a481"
        },
        "date": 1619533243985,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.699939009557602,
            "unit": "iter/sec",
            "range": "stddev: 0.006494050685196185",
            "extra": "mean: 149.25509001999558 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 272.0106933009142,
            "unit": "iter/sec",
            "range": "stddev: 0.00022573557525414978",
            "extra": "mean: 3.676326058599989 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 591.4602179438444,
            "unit": "iter/sec",
            "range": "stddev: 0.000010091548538066905",
            "extra": "mean: 1.6907307874000481 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26100.47252115357,
            "unit": "iter/sec",
            "range": "stddev: 2.3292015648615856e-7",
            "extra": "mean: 38.31348260800769 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 27098.96863656353,
            "unit": "iter/sec",
            "range": "stddev: 1.9501043645950478e-7",
            "extra": "mean: 36.901773399993544 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 23700.674251077544,
            "unit": "iter/sec",
            "range": "stddev: 2.1943356724515388e-7",
            "extra": "mean: 42.19289246399967 usec\nrounds: 250"
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
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "6171f0eb25645b613c29b41df05271267f9ff53e",
          "message": "fix typo",
          "timestamp": "2021-04-28T08:27:46+02:00",
          "tree_id": "b7fd5d3c413260d68ba7cd27a10411c3a8b89f70",
          "url": "https://github.com/histolab/histolab/commit/6171f0eb25645b613c29b41df05271267f9ff53e"
        },
        "date": 1619591819150,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.545871220416961,
            "unit": "iter/sec",
            "range": "stddev: 0.0033110666889124975",
            "extra": "mean: 132.5228023099953 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 319.1009992361775,
            "unit": "iter/sec",
            "range": "stddev: 0.00007745256851875374",
            "extra": "mean: 3.1338040382000374 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 609.4509133067007,
            "unit": "iter/sec",
            "range": "stddev: 0.00001949680542994106",
            "extra": "mean: 1.6408212346000028 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 27302.92567069452,
            "unit": "iter/sec",
            "range": "stddev: 6.829646088367471e-7",
            "extra": "mean: 36.626111503989705 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29025.992552707412,
            "unit": "iter/sec",
            "range": "stddev: 9.871315743397502e-7",
            "extra": "mean: 34.45187957600865 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 25962.806788385118,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012377908954817544",
            "extra": "mean: 38.5166368240034 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "0d217ec8607bf1b78bb37d6d0d12a92931adf09e",
          "message": "release: prepare v0.2.5 release",
          "timestamp": "2021-04-28T15:58:08+02:00",
          "tree_id": "61eb0911d6066ab3587d0f104ee40fd9841df244",
          "url": "https://github.com/histolab/histolab/commit/0d217ec8607bf1b78bb37d6d0d12a92931adf09e"
        },
        "date": 1619618914840,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.459981736589391,
            "unit": "iter/sec",
            "range": "stddev: 0.0035481746178902453",
            "extra": "mean: 118.20356487000481 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 262.5800499124378,
            "unit": "iter/sec",
            "range": "stddev: 0.00010459701746341572",
            "extra": "mean: 3.8083624416000705 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 515.2430254573992,
            "unit": "iter/sec",
            "range": "stddev: 0.000033884890797186594",
            "extra": "mean: 1.9408317057999283 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 18278.535961464873,
            "unit": "iter/sec",
            "range": "stddev: 0.000002865587364658936",
            "extra": "mean: 54.70897680800135 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 19337.29875470582,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025838978961500247",
            "extra": "mean: 51.71353107199866 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 17672.32123041984,
            "unit": "iter/sec",
            "range": "stddev: 0.000003078674869683912",
            "extra": "mean: 56.585662232003415 usec\nrounds: 250"
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
          "id": "22281eee641a4fd3c50b504f37eca93e4b04713e",
          "message": "adress comments",
          "timestamp": "2021-04-29T18:42:12+02:00",
          "tree_id": "94292bbea728b80ffe383eaa0569dc16df23a672",
          "url": "https://github.com/histolab/histolab/commit/22281eee641a4fd3c50b504f37eca93e4b04713e"
        },
        "date": 1619715181607,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.897507198706549,
            "unit": "iter/sec",
            "range": "stddev: 0.001545891521315689",
            "extra": "mean: 144.97991392999552 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 236.5413700586117,
            "unit": "iter/sec",
            "range": "stddev: 0.00009464247196262203",
            "extra": "mean: 4.227590293199933 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 563.3681964742776,
            "unit": "iter/sec",
            "range": "stddev: 0.000024950790077562327",
            "extra": "mean: 1.775038076800024 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26494.398448832533,
            "unit": "iter/sec",
            "range": "stddev: 1.1769616390524082e-7",
            "extra": "mean: 37.74382731999958 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 27492.71481107686,
            "unit": "iter/sec",
            "range": "stddev: 1.4061532956949031e-7",
            "extra": "mean: 36.37327222399654 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24033.35966292255,
            "unit": "iter/sec",
            "range": "stddev: 1.436078520716125e-7",
            "extra": "mean: 41.60883097600163 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "acc0f8206d64a14804099bcac4649c174f6b7fcd",
          "message": "Update numpy requirement from <1.20.3,>=1.18.4 to >=1.18.4,<1.20.4\n\nUpdates the requirements on [numpy](https://github.com/numpy/numpy) to permit the latest version.\n- [Release notes](https://github.com/numpy/numpy/releases)\n- [Changelog](https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst.txt)\n- [Commits](https://github.com/numpy/numpy/compare/v1.18.4...v1.20.3)\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-05-18T20:06:11+02:00",
          "tree_id": "b0b7d20cd87dd86722d11246ec436eec8d6adba5",
          "url": "https://github.com/histolab/histolab/commit/acc0f8206d64a14804099bcac4649c174f6b7fcd"
        },
        "date": 1621361867249,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.36363645721492,
            "unit": "iter/sec",
            "range": "stddev: 0.00624374112077349",
            "extra": "mean: 135.8024674099977 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 230.96949107912118,
            "unit": "iter/sec",
            "range": "stddev: 0.0002813428098290394",
            "extra": "mean: 4.329576150200023 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 530.3461240273659,
            "unit": "iter/sec",
            "range": "stddev: 0.000031748351097860056",
            "extra": "mean: 1.8855610603998685 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17315.381817819427,
            "unit": "iter/sec",
            "range": "stddev: 0.000003036048963482304",
            "extra": "mean: 57.752119503994436 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18557.047025195985,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023933017897000745",
            "extra": "mean: 53.8878841359965 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16444.93289403423,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032802763620188526",
            "extra": "mean: 60.80900459999884 usec\nrounds: 250"
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
          "id": "244bf0dacbc7487b3040371542596de3af89fcb3",
          "message": "Remove 'array' suffix from method name",
          "timestamp": "2021-05-20T11:25:24+02:00",
          "tree_id": "f205b8e5ddf093683cc8809c927df9e02c9c3578",
          "url": "https://github.com/histolab/histolab/commit/244bf0dacbc7487b3040371542596de3af89fcb3"
        },
        "date": 1621503261815,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.765604745640019,
            "unit": "iter/sec",
            "range": "stddev: 0.0016222460943827169",
            "extra": "mean: 114.08226004000426 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 292.67856081325283,
            "unit": "iter/sec",
            "range": "stddev: 0.00008180081654900302",
            "extra": "mean: 3.4167176346000363 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 666.6376257095421,
            "unit": "iter/sec",
            "range": "stddev: 0.000009809683706809765",
            "extra": "mean: 1.5000653450000527 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29696.251426645056,
            "unit": "iter/sec",
            "range": "stddev: 4.770588062701353e-7",
            "extra": "mean: 33.67428385600033 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 31051.324450497836,
            "unit": "iter/sec",
            "range": "stddev: 8.947733043017302e-8",
            "extra": "mean: 32.20474545600155 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27280.639209667064,
            "unit": "iter/sec",
            "range": "stddev: 9.625218575861408e-8",
            "extra": "mean: 36.65603259199452 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "f3634c2427cc8f9bc759fd1e9e6f558f280193e2",
          "message": "delete artifact depends on build",
          "timestamp": "2021-06-03T16:54:38+02:00",
          "tree_id": "5e5af3d34b6b32186cf70fad81bc0ec9935d348b",
          "url": "https://github.com/histolab/histolab/commit/f3634c2427cc8f9bc759fd1e9e6f558f280193e2"
        },
        "date": 1622732656717,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.691389789390517,
            "unit": "iter/sec",
            "range": "stddev: 0.001680783175272894",
            "extra": "mean: 115.05639767999924 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 278.94133089626763,
            "unit": "iter/sec",
            "range": "stddev: 0.0001899217551318885",
            "extra": "mean: 3.584983253599944 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 631.5162542902608,
            "unit": "iter/sec",
            "range": "stddev: 0.00004468783435854109",
            "extra": "mean: 1.5834905170000184 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 30155.535426931157,
            "unit": "iter/sec",
            "range": "stddev: 3.1606883606197836e-7",
            "extra": "mean: 33.16140754400021 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 31090.6996649568,
            "unit": "iter/sec",
            "range": "stddev: 8.858520935284825e-8",
            "extra": "mean: 32.163959343994065 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27509.653908933473,
            "unit": "iter/sec",
            "range": "stddev: 1.047054925045112e-7",
            "extra": "mean: 36.3508753439919 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "44efd1e9787f5ae14ca1c7c6377dc2a0a81683b3",
          "message": "Update scipy requirement from <1.6.4,>=1.5.0 to >=1.5.0,<1.7.1\n\nUpdates the requirements on [scipy](https://github.com/scipy/scipy) to permit the latest version.\n- [Release notes](https://github.com/scipy/scipy/releases)\n- [Commits](https://github.com/scipy/scipy/compare/v1.5.0...v1.7.0)\n\n---\nupdated-dependencies:\n- dependency-name: scipy\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-06-21T10:37:32+02:00",
          "tree_id": "b0cf30b004927aa1d78b97a5a75c9b06dd55a75e",
          "url": "https://github.com/histolab/histolab/commit/44efd1e9787f5ae14ca1c7c6377dc2a0a81683b3"
        },
        "date": 1624265291603,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.449804552488944,
            "unit": "iter/sec",
            "range": "stddev: 0.0015548681749126471",
            "extra": "mean: 134.23170943000173 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 259.10629064966116,
            "unit": "iter/sec",
            "range": "stddev: 0.00006152367447875975",
            "extra": "mean: 3.859419998999965 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 581.8744078640088,
            "unit": "iter/sec",
            "range": "stddev: 0.00001470127474626863",
            "extra": "mean: 1.7185839186000296 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26532.801604330947,
            "unit": "iter/sec",
            "range": "stddev: 7.31814207014313e-7",
            "extra": "mean: 37.68919750399709 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 27197.120807838637,
            "unit": "iter/sec",
            "range": "stddev: 7.681994609860827e-7",
            "extra": "mean: 36.76859793599124 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24260.09212472019,
            "unit": "iter/sec",
            "range": "stddev: 3.610141669232512e-7",
            "extra": "mean: 41.219958887997564 usec\nrounds: 250"
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
          "id": "dec10c6645e65ea81b13efbfb0e913f3058ea662",
          "message": "Put type hints in description and not in signature",
          "timestamp": "2021-06-24T14:18:52+02:00",
          "tree_id": "7f437482e32d48367700ac07dbee94b188c26750",
          "url": "https://github.com/histolab/histolab/commit/dec10c6645e65ea81b13efbfb0e913f3058ea662"
        },
        "date": 1624537713604,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.709712385215707,
            "unit": "iter/sec",
            "range": "stddev: 0.0021922687447684607",
            "extra": "mean: 129.70652471000335 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 295.6538195396826,
            "unit": "iter/sec",
            "range": "stddev: 0.00005776399390459745",
            "extra": "mean: 3.3823341148000283 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 610.6677882266201,
            "unit": "iter/sec",
            "range": "stddev: 0.00002021665374152289",
            "extra": "mean: 1.637551577599993 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 27608.261060939116,
            "unit": "iter/sec",
            "range": "stddev: 9.585117900234512e-7",
            "extra": "mean: 36.22104259999287 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 28328.728114825677,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010177082385603191",
            "extra": "mean: 35.299855183990985 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24956.93310333006,
            "unit": "iter/sec",
            "range": "stddev: 7.816832927481023e-7",
            "extra": "mean: 40.069025943999804 usec\nrounds: 250"
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
          "id": "8dae65b223e420f03f73017d4143ef251ca8add7",
          "message": "Update all lazyproperty docstrings according to CR",
          "timestamp": "2021-06-29T20:16:05+02:00",
          "tree_id": "e6ee49be33746b28ab0fd035f86f6452c7de5d55",
          "url": "https://github.com/histolab/histolab/commit/8dae65b223e420f03f73017d4143ef251ca8add7"
        },
        "date": 1624991111586,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.701753257631943,
            "unit": "iter/sec",
            "range": "stddev: 0.008393461918185515",
            "extra": "mean: 114.91936973999373 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 298.01129780557926,
            "unit": "iter/sec",
            "range": "stddev: 0.0001236530309929793",
            "extra": "mean: 3.3555774810000454 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 671.0720421170181,
            "unit": "iter/sec",
            "range": "stddev: 0.00005709869663277048",
            "extra": "mean: 1.4901529750000009 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 33220.62146976584,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019812485972826407",
            "extra": "mean: 30.10178484800781 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 33951.59437478957,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023180349140625135",
            "extra": "mean: 29.453697783999814 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30484.19668675413,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020167105039249983",
            "extra": "mean: 32.80388229598702 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "50dfce7664993495de90bd2dbe579dc8bbf70f86",
          "message": "Update numpy requirement from <1.20.4,>=1.18.4 to >=1.18.4,<1.21.1\n\nUpdates the requirements on [numpy](https://github.com/numpy/numpy) to permit the latest version.\n- [Release notes](https://github.com/numpy/numpy/releases)\n- [Changelog](https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst.txt)\n- [Commits](https://github.com/numpy/numpy/compare/v1.18.4...v1.21.0)\n\n---\nupdated-dependencies:\n- dependency-name: numpy\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-06-29T20:40:43+02:00",
          "tree_id": "658b0f2f86f4d2e5327a0d466a77dcb92db6b6da",
          "url": "https://github.com/histolab/histolab/commit/50dfce7664993495de90bd2dbe579dc8bbf70f86"
        },
        "date": 1624992694492,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.982231744986079,
            "unit": "iter/sec",
            "range": "stddev: 0.003424461709067292",
            "extra": "mean: 125.27824698000472 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 205.11527764842108,
            "unit": "iter/sec",
            "range": "stddev: 0.00011036874414123818",
            "extra": "mean: 4.875307249000025 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 475.4905843910747,
            "unit": "iter/sec",
            "range": "stddev: 0.00005610659847105713",
            "extra": "mean: 2.1030910660000246 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16327.158959157938,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024965154129928414",
            "extra": "mean: 61.247642808003555 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 16794.10366167817,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034090617022685805",
            "extra": "mean: 59.544708080006785 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15009.088655176434,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028522370747029573",
            "extra": "mean: 66.62629710399597 usec\nrounds: 250"
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
          "id": "22cc4fa4674b4aac93595b76deb931285d32cd46",
          "message": "Add DAB filter",
          "timestamp": "2021-07-01T12:44:37+02:00",
          "tree_id": "3ad8a1dee0de297205e757b876b9f61a069d35c3",
          "url": "https://github.com/histolab/histolab/commit/22cc4fa4674b4aac93595b76deb931285d32cd46"
        },
        "date": 1625136811668,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.064662820696345,
            "unit": "iter/sec",
            "range": "stddev: 0.001254985607123801",
            "extra": "mean: 123.99774450999985 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 278.788276490678,
            "unit": "iter/sec",
            "range": "stddev: 0.000050963778742683335",
            "extra": "mean: 3.58695140479997 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 578.6976691060017,
            "unit": "iter/sec",
            "range": "stddev: 0.00009784133496988836",
            "extra": "mean: 1.7280180194000876 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29515.933219416725,
            "unit": "iter/sec",
            "range": "stddev: 2.2437719108403726e-7",
            "extra": "mean: 33.88000618398746 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 30281.21272230328,
            "unit": "iter/sec",
            "range": "stddev: 7.682023645835224e-8",
            "extra": "mean: 33.02377646399418 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27174.162614532772,
            "unit": "iter/sec",
            "range": "stddev: 1.02164339669434e-7",
            "extra": "mean: 36.79966202399919 usec\nrounds: 250"
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
          "id": "517812044e8e0dbb97f39c6394445af536c4514b",
          "message": "Docstring improvements",
          "timestamp": "2021-07-06T15:48:11+02:00",
          "tree_id": "4af85c0af8c4b966a7fe5ab8e310337495140deb",
          "url": "https://github.com/histolab/histolab/commit/517812044e8e0dbb97f39c6394445af536c4514b"
        },
        "date": 1625580019553,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.283867666503928,
            "unit": "iter/sec",
            "range": "stddev: 0.005179894405882186",
            "extra": "mean: 137.28969906999623 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 209.09400337802415,
            "unit": "iter/sec",
            "range": "stddev: 0.0002099968813935941",
            "extra": "mean: 4.782537919999959 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 456.3867469173675,
            "unit": "iter/sec",
            "range": "stddev: 0.00014261754205849338",
            "extra": "mean: 2.1911240998000725 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 14706.49410157112,
            "unit": "iter/sec",
            "range": "stddev: 0.000006669023715544939",
            "extra": "mean: 67.9971713920022 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 15340.463796573182,
            "unit": "iter/sec",
            "range": "stddev: 0.000006517581121010867",
            "extra": "mean: 65.18707734399686 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 14436.848408236316,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035475134403882045",
            "extra": "mean: 69.26719542400224 usec\nrounds: 250"
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
          "id": "f04b8a19fe5055f3a019467ec37e64549dae18c5",
          "message": "Use correct Windows path for dll",
          "timestamp": "2021-07-13T14:38:05+02:00",
          "tree_id": "af90575788d78df38275c23ab3f38f16dad87002",
          "url": "https://github.com/histolab/histolab/commit/f04b8a19fe5055f3a019467ec37e64549dae18c5"
        },
        "date": 1626180429903,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.6506868880714585,
            "unit": "iter/sec",
            "range": "stddev: 0.00101956255702947",
            "extra": "mean: 130.7072181400008 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 276.737905303154,
            "unit": "iter/sec",
            "range": "stddev: 0.00008355005802695227",
            "extra": "mean: 3.613527387599993 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 590.5399091834452,
            "unit": "iter/sec",
            "range": "stddev: 0.0000979866112903312",
            "extra": "mean: 1.6933656548000728 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29399.950491439136,
            "unit": "iter/sec",
            "range": "stddev: 1.0897786249276867e-7",
            "extra": "mean: 34.013662719982676 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 30135.8468162989,
            "unit": "iter/sec",
            "range": "stddev: 8.408860502451426e-8",
            "extra": "mean: 33.18307283998911 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27086.2857118396,
            "unit": "iter/sec",
            "range": "stddev: 9.289214843196858e-8",
            "extra": "mean: 36.91905234400201 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mtageld@emory.edu",
            "name": "kheffah",
            "username": "kheffah"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "b96805a3a39bc5574c39d4242df38ab5f6d9464b",
          "message": "formatting pep8",
          "timestamp": "2021-07-13T21:55:03+02:00",
          "tree_id": "91180d1d3e8ef9000244bd8788512722fbeb1e92",
          "url": "https://github.com/histolab/histolab/commit/b96805a3a39bc5574c39d4242df38ab5f6d9464b"
        },
        "date": 1626206731209,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.324244893623435,
            "unit": "iter/sec",
            "range": "stddev: 0.005071628345565893",
            "extra": "mean: 120.13101642000265 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 212.36768013536891,
            "unit": "iter/sec",
            "range": "stddev: 0.0001764719125887393",
            "extra": "mean: 4.708814445600069 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 442.3586641260365,
            "unit": "iter/sec",
            "range": "stddev: 0.00011281758452444094",
            "extra": "mean: 2.2606090511998675 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17756.983546323267,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026606070313765123",
            "extra": "mean: 56.315871296003934 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18414.878309357653,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031642511760375166",
            "extra": "mean: 54.30391573599718 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16439.41726111903,
            "unit": "iter/sec",
            "range": "stddev: 0.000003877934200738622",
            "extra": "mean: 60.829406791997826 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "c551867845828592f98b7ee75bd3d8a5c06185bc",
          "message": "pin black and flake8 version in ci workflow to be compliant with pre-commit",
          "timestamp": "2021-07-13T22:17:51+02:00",
          "tree_id": "f7d71b3c0004ae792f2bb926e398f31ba3455b1d",
          "url": "https://github.com/histolab/histolab/commit/c551867845828592f98b7ee75bd3d8a5c06185bc"
        },
        "date": 1626208066641,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.32013631941978,
            "unit": "iter/sec",
            "range": "stddev: 0.0070034034533320855",
            "extra": "mean: 107.29456798999422 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 268.7530843232731,
            "unit": "iter/sec",
            "range": "stddev: 0.00030290094669796773",
            "extra": "mean: 3.7208875295999864 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 489.3086780470187,
            "unit": "iter/sec",
            "range": "stddev: 0.00010411701002157991",
            "extra": "mean: 2.0436997029999704 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16793.33031127958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023976429578247755",
            "extra": "mean: 59.547450175998165 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 17098.319833299716,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019315602464712996",
            "extra": "mean: 58.485278656003175 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 18026.301560252203,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028297384051205464",
            "extra": "mean: 55.47449634399709 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "a97bec98605d412d361c536015b9c6c1d0e03c40",
          "message": "tweak in contributing md",
          "timestamp": "2021-07-13T22:36:16+02:00",
          "tree_id": "c117e75ccd58e6f1f9bb2306fd6c39c5683c3d1e",
          "url": "https://github.com/histolab/histolab/commit/a97bec98605d412d361c536015b9c6c1d0e03c40"
        },
        "date": 1626209227534,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.033763001529143,
            "unit": "iter/sec",
            "range": "stddev: 0.004590948442988046",
            "extra": "mean: 124.47467019000442 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 252.84938982508402,
            "unit": "iter/sec",
            "range": "stddev: 0.00022488572496919316",
            "extra": "mean: 3.954923524600076 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 458.67476758518455,
            "unit": "iter/sec",
            "range": "stddev: 0.00012585840884688556",
            "extra": "mean: 2.180194051800072 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16372.548275304027,
            "unit": "iter/sec",
            "range": "stddev: 0.000001662598784708507",
            "extra": "mean: 61.077847088005 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 16872.65593092911,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035329632449855135",
            "extra": "mean: 59.267491975991106 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16002.286236647022,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026684558815536227",
            "extra": "mean: 62.49107066400848 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "21d676230b47345069b64366ec288f11d774f804",
          "message": "release: prepare v0.2.6 release",
          "timestamp": "2021-07-13T22:41:34+02:00",
          "tree_id": "c5fa1d688f44127179a801df2e99bb36aa696a9b",
          "url": "https://github.com/histolab/histolab/commit/21d676230b47345069b64366ec288f11d774f804"
        },
        "date": 1626209633850,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.739220753568176,
            "unit": "iter/sec",
            "range": "stddev: 0.005967877366799406",
            "extra": "mean: 129.21197519000202 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 235.43683540162633,
            "unit": "iter/sec",
            "range": "stddev: 0.00029741826488124",
            "extra": "mean: 4.2474237231999945 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 457.29461017631263,
            "unit": "iter/sec",
            "range": "stddev: 0.00012569143862240775",
            "extra": "mean: 2.1867740790000654 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16590.162261526635,
            "unit": "iter/sec",
            "range": "stddev: 0.000002125207239980759",
            "extra": "mean: 60.27668592000737 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 17595.5849295626,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028530196513687633",
            "extra": "mean: 56.83243859201775 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16013.362373545287,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032558668224940296",
            "extra": "mean: 62.447846784010835 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "dd2110f3b163ac2f37427b337cd8a64638471994",
          "message": "requirements for docs",
          "timestamp": "2021-07-13T22:53:35+02:00",
          "tree_id": "5dea0763c7b5e06022924ab8fb79673d368744f1",
          "url": "https://github.com/histolab/histolab/commit/dd2110f3b163ac2f37427b337cd8a64638471994"
        },
        "date": 1626210179401,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.562710312871187,
            "unit": "iter/sec",
            "range": "stddev: 0.0013594821004055823",
            "extra": "mean: 132.2277277100079 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 209.00928093919953,
            "unit": "iter/sec",
            "range": "stddev: 0.00026746878418228067",
            "extra": "mean: 4.7844765337999435 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 579.8024464519735,
            "unit": "iter/sec",
            "range": "stddev: 0.00009944600393856072",
            "extra": "mean: 1.7247253889999457 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29143.794054616603,
            "unit": "iter/sec",
            "range": "stddev: 9.803155937305952e-8",
            "extra": "mean: 34.312622376000945 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29867.20932407294,
            "unit": "iter/sec",
            "range": "stddev: 5.640985360912077e-7",
            "extra": "mean: 33.481534519999514 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26755.57501721012,
            "unit": "iter/sec",
            "range": "stddev: 1.208289884977093e-7",
            "extra": "mean: 37.375388096004826 usec\nrounds: 250"
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
          "id": "dc87eeedf6abf122e25153543d54a4dc6254432f",
          "message": "Update numpy requirement from <1.21.1,>=1.18.4 to >=1.18.4,<1.21.2\n\nUpdates the requirements on [numpy](https://github.com/numpy/numpy) to permit the latest version.\n- [Release notes](https://github.com/numpy/numpy/releases)\n- [Changelog](https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst.txt)\n- [Commits](https://github.com/numpy/numpy/compare/v1.18.4...v1.21.1)\n\n---\nupdated-dependencies:\n- dependency-name: numpy\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-07-19T11:13:52+02:00",
          "tree_id": "dbfc5754828ddbe8432804ab9bc1a3b28e8f497b",
          "url": "https://github.com/histolab/histolab/commit/dc87eeedf6abf122e25153543d54a4dc6254432f"
        },
        "date": 1626686663708,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.032844813933253,
            "unit": "iter/sec",
            "range": "stddev: 0.001591404288548698",
            "extra": "mean: 142.18997097999818 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 168.17138567847033,
            "unit": "iter/sec",
            "range": "stddev: 0.00006308456214417532",
            "extra": "mean: 5.946314802399956 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 513.3776524163153,
            "unit": "iter/sec",
            "range": "stddev: 0.00011177815736499572",
            "extra": "mean: 1.9478837757999372 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 25283.6118414807,
            "unit": "iter/sec",
            "range": "stddev: 2.498789204966826e-7",
            "extra": "mean: 39.5513111919945 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 26759.537784734053,
            "unit": "iter/sec",
            "range": "stddev: 2.7058721387872e-7",
            "extra": "mean: 37.36985324800662 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 22922.705920594795,
            "unit": "iter/sec",
            "range": "stddev: 2.9144601137364975e-7",
            "extra": "mean: 43.62486712799273 usec\nrounds: 250"
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
          "id": "cd32904484c95d94274df5b7d276328bcb008ad4",
          "message": "Link automatically issues",
          "timestamp": "2021-07-20T14:21:54+02:00",
          "tree_id": "b3f9c29e78cf7feef0d26963a8a2e1eb67c8a182",
          "url": "https://github.com/histolab/histolab/commit/cd32904484c95d94274df5b7d276328bcb008ad4"
        },
        "date": 1626784458483,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.9731836436123125,
            "unit": "iter/sec",
            "range": "stddev: 0.005069385113434302",
            "extra": "mean: 143.40652004999697 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 185.95086733744088,
            "unit": "iter/sec",
            "range": "stddev: 0.00010473715430929216",
            "extra": "mean: 5.3777646446000285 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 444.31445833417905,
            "unit": "iter/sec",
            "range": "stddev: 0.00013612671518728256",
            "extra": "mean: 2.2506582471999534 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15471.771531655226,
            "unit": "iter/sec",
            "range": "stddev: 0.000004014245295790794",
            "extra": "mean: 64.6338396320034 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 16466.807723741666,
            "unit": "iter/sec",
            "range": "stddev: 0.000004734367642925164",
            "extra": "mean: 60.72822472799089 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 14981.653137372816,
            "unit": "iter/sec",
            "range": "stddev: 0.000003910255885759229",
            "extra": "mean: 66.74830813599783 usec\nrounds: 250"
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
          "id": "5951a6ba7a81d74b86c6c456520de13a7d9ee430",
          "message": "Add RGB to OD filter",
          "timestamp": "2021-07-21T16:11:25+02:00",
          "tree_id": "02060d0c0bd5376655cfed5b77628efa86b491f6",
          "url": "https://github.com/histolab/histolab/commit/5951a6ba7a81d74b86c6c456520de13a7d9ee430"
        },
        "date": 1626877266898,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.36854406660547,
            "unit": "iter/sec",
            "range": "stddev: 0.005722458954298992",
            "extra": "mean: 106.74017145999642 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 308.5538918577334,
            "unit": "iter/sec",
            "range": "stddev: 0.00021960416886256088",
            "extra": "mean: 3.240924928800041 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 489.84504490689693,
            "unit": "iter/sec",
            "range": "stddev: 0.00011182274122327354",
            "extra": "mean: 2.041461908000042 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16094.48782194119,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017844553090581958",
            "extra": "mean: 62.13307382399125 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 17111.262691530093,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016528199222089371",
            "extra": "mean: 58.44104073599374 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 17164.10955970643,
            "unit": "iter/sec",
            "range": "stddev: 0.000003145408596971684",
            "extra": "mean: 58.261105624002084 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "0b7aa0943a4ea00a75278bd4797f44f1a4bc2fb1",
          "message": "Include histolab version",
          "timestamp": "2021-07-21T22:41:46+02:00",
          "tree_id": "c91d11f3532e8c250f71dc001cebe9ba66f0c499",
          "url": "https://github.com/histolab/histolab/commit/0b7aa0943a4ea00a75278bd4797f44f1a4bc2fb1"
        },
        "date": 1626900649020,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.327043659118987,
            "unit": "iter/sec",
            "range": "stddev: 0.0013162060798967133",
            "extra": "mean: 120.09063972000376 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 197.43149760531404,
            "unit": "iter/sec",
            "range": "stddev: 0.00004106077277104057",
            "extra": "mean: 5.065047938800035 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 595.9809314050889,
            "unit": "iter/sec",
            "range": "stddev: 0.0000963299503696485",
            "extra": "mean: 1.6779060324000512 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29407.727191005066,
            "unit": "iter/sec",
            "range": "stddev: 1.0142742816168292e-7",
            "extra": "mean: 34.00466800800132 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 30407.952326998602,
            "unit": "iter/sec",
            "range": "stddev: 8.892977854730813e-8",
            "extra": "mean: 32.88613416800581 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26861.29540723744,
            "unit": "iter/sec",
            "range": "stddev: 1.048919190265331e-7",
            "extra": "mean: 37.22828645600475 usec\nrounds: 250"
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
          "id": "d2a0bf09bde4b3e60e34dfa9ffbc8433d098f8c6",
          "message": "Update scipy requirement from <1.7.1,>=1.5.0 to >=1.5.0,<1.7.2\n\nUpdates the requirements on [scipy](https://github.com/scipy/scipy) to permit the latest version.\n- [Release notes](https://github.com/scipy/scipy/releases)\n- [Commits](https://github.com/scipy/scipy/compare/v1.5.0...v1.7.1)\n\n---\nupdated-dependencies:\n- dependency-name: scipy\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-08-02T11:06:10+02:00",
          "tree_id": "8f5eada24d1714e1d7eadb6aa0492171795cc466",
          "url": "https://github.com/histolab/histolab/commit/d2a0bf09bde4b3e60e34dfa9ffbc8433d098f8c6"
        },
        "date": 1627895836632,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.834752821352755,
            "unit": "iter/sec",
            "range": "stddev: 0.005762337200521884",
            "extra": "mean: 127.63644531000523 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 211.76580054711124,
            "unit": "iter/sec",
            "range": "stddev: 0.0001433415762459476",
            "extra": "mean: 4.722197812000013 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 435.1679885849175,
            "unit": "iter/sec",
            "range": "stddev: 0.00009568652702667979",
            "extra": "mean: 2.2979631457998724 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15828.053343977592,
            "unit": "iter/sec",
            "range": "stddev: 0.000005371180829395956",
            "extra": "mean: 63.1789632159971 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 16282.450912131755,
            "unit": "iter/sec",
            "range": "stddev: 0.000004520519630597698",
            "extra": "mean: 61.415815431994815 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 14892.944076690332,
            "unit": "iter/sec",
            "range": "stddev: 0.00000419394329446203",
            "extra": "mean: 67.14589102400168 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mtageld@emory.edu",
            "name": "kheffah",
            "username": "kheffah"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "ffd63f39c7f358b108b3a05dd15f090b1888ad97",
          "message": "fix test case",
          "timestamp": "2021-08-10T12:52:25+02:00",
          "tree_id": "c57a5e5c535e1de4c4320d332b2e8a788e3a2fca",
          "url": "https://github.com/histolab/histolab/commit/ffd63f39c7f358b108b3a05dd15f090b1888ad97"
        },
        "date": 1628593298509,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.844481594783074,
            "unit": "iter/sec",
            "range": "stddev: 0.001044764451199392",
            "extra": "mean: 127.47814982000136 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 166.50105843685728,
            "unit": "iter/sec",
            "range": "stddev: 0.00010563759017531539",
            "extra": "mean: 6.005967826199935 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 585.7701911510003,
            "unit": "iter/sec",
            "range": "stddev: 0.00009879769014822372",
            "extra": "mean: 1.7071541282001135 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29849.382435222444,
            "unit": "iter/sec",
            "range": "stddev: 1.058294107059382e-7",
            "extra": "mean: 33.50153063200378 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 30639.806093889074,
            "unit": "iter/sec",
            "range": "stddev: 8.091162968189275e-8",
            "extra": "mean: 32.63728226398416 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26922.508041366356,
            "unit": "iter/sec",
            "range": "stddev: 2.25327993967735e-7",
            "extra": "mean: 37.14364198400472 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "86db1a16fff5550940a66c66714a1ba8c78d9f20",
          "message": "address Alessias's comment",
          "timestamp": "2021-08-12T15:50:46+02:00",
          "tree_id": "b2f134b6cd84bd929d973ad23c93178391771d82",
          "url": "https://github.com/histolab/histolab/commit/86db1a16fff5550940a66c66714a1ba8c78d9f20"
        },
        "date": 1628776900853,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.92851134567859,
            "unit": "iter/sec",
            "range": "stddev: 0.0031534559772878365",
            "extra": "mean: 126.1270819199933 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 218.14874332290816,
            "unit": "iter/sec",
            "range": "stddev: 0.00011089071715729637",
            "extra": "mean: 4.584028240400085 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 440.4563030152137,
            "unit": "iter/sec",
            "range": "stddev: 0.00009188224818176161",
            "extra": "mean: 2.2703727774000297 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17785.853882187632,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026598378592994195",
            "extra": "mean: 56.22445830399465 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18156.670896838365,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029395472998947587",
            "extra": "mean: 55.07617589599704 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16210.115332346222,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016778354921058129",
            "extra": "mean: 61.689875704003505 usec\nrounds: 250"
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
          "id": "79e9792d0ff5cf2c5a19dec12c7b06c202daba49",
          "message": "Update numpy requirement from <1.21.2,>=1.18.4 to >=1.18.4,<1.21.3\n\nUpdates the requirements on [numpy](https://github.com/numpy/numpy) to permit the latest version.\n- [Release notes](https://github.com/numpy/numpy/releases)\n- [Changelog](https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst.txt)\n- [Commits](https://github.com/numpy/numpy/compare/v1.18.4...v1.21.2)\n\n---\nupdated-dependencies:\n- dependency-name: numpy\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-08-16T20:23:07+02:00",
          "tree_id": "bb24a9adb347065168613b3fb62c1456c8acd70d",
          "url": "https://github.com/histolab/histolab/commit/79e9792d0ff5cf2c5a19dec12c7b06c202daba49"
        },
        "date": 1629138971254,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.576976063933412,
            "unit": "iter/sec",
            "range": "stddev: 0.010015063153256372",
            "extra": "mean: 152.04555867000408 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 162.69798841575712,
            "unit": "iter/sec",
            "range": "stddev: 0.0002776600282997874",
            "extra": "mean: 6.146357491800131 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 408.3194992795035,
            "unit": "iter/sec",
            "range": "stddev: 0.0001786471486874536",
            "extra": "mean: 2.4490625644000374 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 14279.219980603202,
            "unit": "iter/sec",
            "range": "stddev: 0.000005759914551928085",
            "extra": "mean: 70.03183656799138 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 15028.863322762956,
            "unit": "iter/sec",
            "range": "stddev: 0.0000060326698781527735",
            "extra": "mean: 66.53863159999491 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 13393.245909562518,
            "unit": "iter/sec",
            "range": "stddev: 0.0000070251673118879165",
            "extra": "mean: 74.66449931200168 usec\nrounds: 250"
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
          "id": "f86c65e81f54d258c8805d3078c49ead89d6619e",
          "message": "Add method dispatcher compatible with older Python versions",
          "timestamp": "2021-08-18T15:31:43+02:00",
          "tree_id": "104cf43ec690f81ed65370fce1c480ac00b8a482",
          "url": "https://github.com/histolab/histolab/commit/f86c65e81f54d258c8805d3078c49ead89d6619e"
        },
        "date": 1629294001736,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.112212802444578,
            "unit": "iter/sec",
            "range": "stddev: 0.003231281773039168",
            "extra": "mean: 109.74282774999779 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 224.27920855271546,
            "unit": "iter/sec",
            "range": "stddev: 0.00007836702583377333",
            "extra": "mean: 4.458728057999883 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 621.5206551208668,
            "unit": "iter/sec",
            "range": "stddev: 0.00008502335777197446",
            "extra": "mean: 1.6089569859999755 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 35016.21784252577,
            "unit": "iter/sec",
            "range": "stddev: 0.000001203767728437432",
            "extra": "mean: 28.558195648004585 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 35851.250654314106,
            "unit": "iter/sec",
            "range": "stddev: 0.000001301790522719352",
            "extra": "mean: 27.893029720001323 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 32121.068127410068,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012842520777406075",
            "extra": "mean: 31.132215031998385 usec\nrounds: 250"
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
          "id": "a7dc3522e6b626dd0ee6966c1db74e28357719d3",
          "message": "Pin pooch and be compliant with future versions",
          "timestamp": "2021-08-23T20:57:39+02:00",
          "tree_id": "eba7693dc5723097d763ecaf0c8fb28861555e78",
          "url": "https://github.com/histolab/histolab/commit/a7dc3522e6b626dd0ee6966c1db74e28357719d3"
        },
        "date": 1629745633624,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.66815103933023,
            "unit": "iter/sec",
            "range": "stddev: 0.00391369294058224",
            "extra": "mean: 130.40953352000543 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 192.47158290432378,
            "unit": "iter/sec",
            "range": "stddev: 0.00012743325560723483",
            "extra": "mean: 5.195572171799995 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 549.3009183713117,
            "unit": "iter/sec",
            "range": "stddev: 0.00010861729383150127",
            "extra": "mean: 1.8204957730000162 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 27472.616737401928,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017408219832100965",
            "extra": "mean: 36.39988172799622 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29048.827499979878,
            "unit": "iter/sec",
            "range": "stddev: 0.0000020519380739095933",
            "extra": "mean: 34.42479735199959 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26212.141116248837,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014912191706009977",
            "extra": "mean: 38.15026004800893 usec\nrounds: 250"
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
          "id": "6625b404b60de30aef914dd62f2ff95e5a3da73c",
          "message": "Add message to exception",
          "timestamp": "2021-08-23T23:53:46+02:00",
          "tree_id": "eefac2df48ff99114d631c940c7b126b4aca3c60",
          "url": "https://github.com/histolab/histolab/commit/6625b404b60de30aef914dd62f2ff95e5a3da73c"
        },
        "date": 1629756218550,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.742453655851248,
            "unit": "iter/sec",
            "range": "stddev: 0.002611073691452044",
            "extra": "mean: 114.38436385999125 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 236.8672328735117,
            "unit": "iter/sec",
            "range": "stddev: 0.00012698755168596092",
            "extra": "mean: 4.221774315799962 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 462.3138975804845,
            "unit": "iter/sec",
            "range": "stddev: 0.00008714664088485871",
            "extra": "mean: 2.1630325310000216 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 18382.53051912394,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031347715347214306",
            "extra": "mean: 54.39947448800194 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 19263.213309640734,
            "unit": "iter/sec",
            "range": "stddev: 0.000002152095498439817",
            "extra": "mean: 51.91241896799875 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 17422.79202717328,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021926614871747543",
            "extra": "mean: 57.39608200800194 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "fbfc897ee0fe2b0c3b689e3172ab4ee0ec22fe88",
          "message": "address alessia's comments",
          "timestamp": "2021-08-30T22:12:34+02:00",
          "tree_id": "0fc3556f312b2affd7f5037bf8dc8ea0994c0592",
          "url": "https://github.com/histolab/histolab/commit/fbfc897ee0fe2b0c3b689e3172ab4ee0ec22fe88"
        },
        "date": 1630355037254,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.569148312429035,
            "unit": "iter/sec",
            "range": "stddev: 0.005083788180821715",
            "extra": "mean: 132.11526035999782 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 227.99867291238544,
            "unit": "iter/sec",
            "range": "stddev: 0.00013764579621929",
            "extra": "mean: 4.385990441199965 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 422.8754728231972,
            "unit": "iter/sec",
            "range": "stddev: 0.00015290280936926742",
            "extra": "mean: 2.3647623573999454 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16242.851700288318,
            "unit": "iter/sec",
            "range": "stddev: 0.000005657321748533258",
            "extra": "mean: 61.565543935997994 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 16327.232137832847,
            "unit": "iter/sec",
            "range": "stddev: 0.000007169207315714331",
            "extra": "mean: 61.24736829599169 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 14733.8005475704,
            "unit": "iter/sec",
            "range": "stddev: 0.000006041018867718684",
            "extra": "mean: 67.87115088000155 usec\nrounds: 250"
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
          "id": "cc4a8eaba7b963de419661c5dd91db3a6709426c",
          "message": "Remove _ from subfunction",
          "timestamp": "2021-09-05T18:28:27+02:00",
          "tree_id": "031861fe5c44138cc97201d455559009fd42c49c",
          "url": "https://github.com/histolab/histolab/commit/cc4a8eaba7b963de419661c5dd91db3a6709426c"
        },
        "date": 1630859852701,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.073336159589125,
            "unit": "iter/sec",
            "range": "stddev: 0.001442285335421263",
            "extra": "mean: 123.86453137000217 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 176.67273005415097,
            "unit": "iter/sec",
            "range": "stddev: 0.00011167126855132672",
            "extra": "mean: 5.660183094999979 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 597.974073430739,
            "unit": "iter/sec",
            "range": "stddev: 0.0000974130795850514",
            "extra": "mean: 1.6723133065999491 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29794.31203509929,
            "unit": "iter/sec",
            "range": "stddev: 7.631156867401062e-8",
            "extra": "mean: 33.5634532800068 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 30317.79355443588,
            "unit": "iter/sec",
            "range": "stddev: 0.000001509415730206901",
            "extra": "mean: 32.98393064800348 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27062.84576543125,
            "unit": "iter/sec",
            "range": "stddev: 9.573698986192815e-8",
            "extra": "mean: 36.95102904800024 usec\nrounds: 250"
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
          "id": "41ea99cec9a535ad7b65875f3c6cb5c99ae70fcb",
          "message": "Minor fixes",
          "timestamp": "2021-09-21T18:11:32+02:00",
          "tree_id": "7b97a44b195b4c39942888f6adf9eca1bf5934cc",
          "url": "https://github.com/histolab/histolab/commit/41ea99cec9a535ad7b65875f3c6cb5c99ae70fcb"
        },
        "date": 1632241357325,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.817159786963237,
            "unit": "iter/sec",
            "range": "stddev: 0.003445814236833185",
            "extra": "mean: 127.92369956000016 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 205.47963022533838,
            "unit": "iter/sec",
            "range": "stddev: 0.00007723451079074252",
            "extra": "mean: 4.866662446799978 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 430.6173897618178,
            "unit": "iter/sec",
            "range": "stddev: 0.00009883007007043517",
            "extra": "mean: 2.3222471357998757 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17498.90899202104,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035882066301466785",
            "extra": "mean: 57.14641984000082 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 17182.18819825281,
            "unit": "iter/sec",
            "range": "stddev: 0.000003851885283394769",
            "extra": "mean: 58.19980484800453 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15281.362612256029,
            "unit": "iter/sec",
            "range": "stddev: 0.000004023294188949026",
            "extra": "mean: 65.43919056000777 usec\nrounds: 250"
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
          "id": "7bbdefba13ad82008ff5d5be903ea5cbe0e53198",
          "message": "Address comments",
          "timestamp": "2021-09-22T10:36:37+02:00",
          "tree_id": "50e0848802fd029e15e102200dd2d16452e42dcf",
          "url": "https://github.com/histolab/histolab/commit/7bbdefba13ad82008ff5d5be903ea5cbe0e53198"
        },
        "date": 1632300340450,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.23407062861989,
            "unit": "iter/sec",
            "range": "stddev: 0.0012540429600880183",
            "extra": "mean: 121.44661433000238 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 200.11931331189626,
            "unit": "iter/sec",
            "range": "stddev: 0.000058757346839985755",
            "extra": "mean: 4.997018945599962 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 583.8234294769715,
            "unit": "iter/sec",
            "range": "stddev: 0.0000988697087424602",
            "extra": "mean: 1.7128466408000576 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29431.79799588859,
            "unit": "iter/sec",
            "range": "stddev: 8.340144794009451e-8",
            "extra": "mean: 33.97685727999669 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 30537.24788506225,
            "unit": "iter/sec",
            "range": "stddev: 5.898843708345752e-7",
            "extra": "mean: 32.746893359999376 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26899.422124148205,
            "unit": "iter/sec",
            "range": "stddev: 7.833354611332304e-8",
            "extra": "mean: 37.17551980799908 usec\nrounds: 250"
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
          "id": "06bd9cf392def0b2d03e70143cf7913c48c089a9",
          "message": "address comments",
          "timestamp": "2021-09-22T16:07:43+02:00",
          "tree_id": "195c6d4cf259cdc09f8f0042fb11c0a4e5f0b3a0",
          "url": "https://github.com/histolab/histolab/commit/06bd9cf392def0b2d03e70143cf7913c48c089a9"
        },
        "date": 1632320285430,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.185029954907067,
            "unit": "iter/sec",
            "range": "stddev: 0.005495494116444138",
            "extra": "mean: 122.17426270999567 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 221.6484347425972,
            "unit": "iter/sec",
            "range": "stddev: 0.00013751872986216554",
            "extra": "mean: 4.511649275400077 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 444.1048412225784,
            "unit": "iter/sec",
            "range": "stddev: 0.00010979006319859962",
            "extra": "mean: 2.2517205560000093 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17655.703637632003,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029156616959047743",
            "extra": "mean: 56.638920799993684 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18557.24531210065,
            "unit": "iter/sec",
            "range": "stddev: 0.000002013959981716137",
            "extra": "mean: 53.88730833600221 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16114.534554022051,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035122231985203985",
            "extra": "mean: 62.055779312000595 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "06204cc2a9f632e8e734968965230a94328b17d5",
          "message": "add warning in readme",
          "timestamp": "2021-10-05T11:05:36+02:00",
          "tree_id": "557ba3a6746b4cfe792f59dbdb2c43787111feba",
          "url": "https://github.com/histolab/histolab/commit/06204cc2a9f632e8e734968965230a94328b17d5"
        },
        "date": 1633425417822,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.6643103909342685,
            "unit": "iter/sec",
            "range": "stddev: 0.003177765107905844",
            "extra": "mean: 130.4748828000038 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 210.20463523736484,
            "unit": "iter/sec",
            "range": "stddev: 0.00008212783339846718",
            "extra": "mean: 4.75726902439992 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 430.6521457369789,
            "unit": "iter/sec",
            "range": "stddev: 0.0000984017335014944",
            "extra": "mean: 2.3220597178000615 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16927.81750847331,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025699547244746447",
            "extra": "mean: 59.07436085599602 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 17146.870201061574,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027540843125124595",
            "extra": "mean: 58.319680984001934 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15113.42007435625,
            "unit": "iter/sec",
            "range": "stddev: 0.000002535574877625591",
            "extra": "mean: 66.16636043199469 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "5ebbc8ad070a2e0245511b6846a5e4635f75a432",
          "message": "get rid of src directory. Histolab now lives in the histolab child dir within the root",
          "timestamp": "2021-10-06T12:24:32+02:00",
          "tree_id": "2d52167e9a3359ae9495de2ca7b738752371d1ba",
          "url": "https://github.com/histolab/histolab/commit/5ebbc8ad070a2e0245511b6846a5e4635f75a432"
        },
        "date": 1633516428730,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.751524183498192,
            "unit": "iter/sec",
            "range": "stddev: 0.0008919997469581406",
            "extra": "mean: 129.0068864300065 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 172.52017893297204,
            "unit": "iter/sec",
            "range": "stddev: 0.0001195512666686529",
            "extra": "mean: 5.796423387599908 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 581.198442521862,
            "unit": "iter/sec",
            "range": "stddev: 0.00009789135530253264",
            "extra": "mean: 1.7205827249999628 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 28803.641182850977,
            "unit": "iter/sec",
            "range": "stddev: 7.723950103261876e-7",
            "extra": "mean: 34.7178328480004 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29997.165395055832,
            "unit": "iter/sec",
            "range": "stddev: 2.3415777078744054e-7",
            "extra": "mean: 33.33648319200256 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26559.675776543896,
            "unit": "iter/sec",
            "range": "stddev: 1.3113033357593479e-7",
            "extra": "mean: 37.65106202400057 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "5de2c9f0e48f0e1469d9029b91c22b92d55ebd52",
          "message": "Update scikit-image requirement\n\nUpdates the requirements on [scikit-image](https://github.com/scikit-image/scikit-image) to permit the latest version.\n- [Release notes](https://github.com/scikit-image/scikit-image/releases)\n- [Changelog](https://github.com/scikit-image/scikit-image/blob/master/RELEASE.txt)\n- [Commits](https://github.com/scikit-image/scikit-image/compare/v0.17.1...v0.18.0)\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-10-08T17:20:36+02:00",
          "tree_id": "5caedc36e52645c30da309e3df01c443acf31170",
          "url": "https://github.com/histolab/histolab/commit/5de2c9f0e48f0e1469d9029b91c22b92d55ebd52"
        },
        "date": 1633706571990,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.712232485198931,
            "unit": "iter/sec",
            "range": "stddev: 0.0030756231885228366",
            "extra": "mean: 129.66414094999962 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 203.2229966464075,
            "unit": "iter/sec",
            "range": "stddev: 0.00007730761495787745",
            "extra": "mean: 4.920702954400007 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 426.076476638756,
            "unit": "iter/sec",
            "range": "stddev: 0.00009843850430085827",
            "extra": "mean: 2.3469965014000023 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17271.739146984382,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015698459369552066",
            "extra": "mean: 57.89804903199911 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 17820.74354039708,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013924737887893492",
            "extra": "mean: 56.11438140800033 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15577.315424947748,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015963624048620757",
            "extra": "mean: 64.19591391199901 usec\nrounds: 250"
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
          "id": "0b52a6189f6564f064fa594301e452987fada9f4",
          "message": "change watershed test expectations according to skimage 0.18.1 changes",
          "timestamp": "2021-10-08T19:09:49+02:00",
          "tree_id": "5ace570bae66b8b867d02c5ec055fb9d7b0a780a",
          "url": "https://github.com/histolab/histolab/commit/0b52a6189f6564f064fa594301e452987fada9f4"
        },
        "date": 1633713119228,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.382966038725586,
            "unit": "iter/sec",
            "range": "stddev: 0.002910183235345241",
            "extra": "mean: 135.4469185900001 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 175.03170524221227,
            "unit": "iter/sec",
            "range": "stddev: 0.0001063770738998994",
            "extra": "mean: 5.713250628600005 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 521.2553588841672,
            "unit": "iter/sec",
            "range": "stddev: 0.00010658446282475623",
            "extra": "mean: 1.9184455046000182 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26965.47292739958,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013901393471438247",
            "extra": "mean: 37.0844599200002 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 27896.910523681345,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013642982144815845",
            "extra": "mean: 35.84626330399965 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24838.860979350215,
            "unit": "iter/sec",
            "range": "stddev: 0.000001143411098452679",
            "extra": "mean: 40.2594950240009 usec\nrounds: 250"
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
          "id": "0e8147be8ecc02e35cde666f900f7f1c5d957b2a",
          "message": "clarify docstring",
          "timestamp": "2021-10-12T16:11:23+02:00",
          "tree_id": "f5b0a36b31b1fd2b9e75a6eba257c0439677c085",
          "url": "https://github.com/histolab/histolab/commit/0e8147be8ecc02e35cde666f900f7f1c5d957b2a"
        },
        "date": 1634048014256,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.00198536170538,
            "unit": "iter/sec",
            "range": "stddev: 0.005697483928874804",
            "extra": "mean: 124.96898642000019 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 192.9418480080945,
            "unit": "iter/sec",
            "range": "stddev: 0.00012200618458807586",
            "extra": "mean: 5.182908790000016 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 560.5697818717462,
            "unit": "iter/sec",
            "range": "stddev: 0.00011437881778231379",
            "extra": "mean: 1.7838992260000055 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29313.225257499456,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024070093917032807",
            "extra": "mean: 34.11429452800189 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 31624.431025918875,
            "unit": "iter/sec",
            "range": "stddev: 0.000001997843337124158",
            "extra": "mean: 31.621122263999496 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26438.33659059895,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027278798133915953",
            "extra": "mean: 37.82386219999876 usec\nrounds: 250"
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
          "id": "47473c485deefd7a92327aed4594255ad1645293",
          "message": "Add pytest-html to requirements for testing",
          "timestamp": "2021-10-13T19:51:50+02:00",
          "tree_id": "98cdb03915efbd14acbb4b0270137c5f196e2c90",
          "url": "https://github.com/histolab/histolab/commit/47473c485deefd7a92327aed4594255ad1645293"
        },
        "date": 1634147637128,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.738294029617869,
            "unit": "iter/sec",
            "range": "stddev: 0.0009443609875844882",
            "extra": "mean: 129.22744938000008 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 167.12997498504447,
            "unit": "iter/sec",
            "range": "stddev: 0.00008932015992604482",
            "extra": "mean: 5.983367137400006 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 581.0505358825694,
            "unit": "iter/sec",
            "range": "stddev: 0.00009716507759007785",
            "extra": "mean: 1.7210207000000082 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 28958.512554093842,
            "unit": "iter/sec",
            "range": "stddev: 1.0455637430363998e-7",
            "extra": "mean: 34.532160383998416 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29620.491900083038,
            "unit": "iter/sec",
            "range": "stddev: 2.1343256255765473e-7",
            "extra": "mean: 33.760411655999405 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26231.504061559797,
            "unit": "iter/sec",
            "range": "stddev: 1.0609242620053676e-7",
            "extra": "mean: 38.12209920000055 usec\nrounds: 250"
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
          "id": "b8037b71d1809e3bfb72cad8d8c4da584d172a84",
          "message": "kmeans_segmentation always returns a RGB image",
          "timestamp": "2021-10-13T20:29:16+02:00",
          "tree_id": "b239ac7a40be8a6570f526779956bbe0dd8211bd",
          "url": "https://github.com/histolab/histolab/commit/b8037b71d1809e3bfb72cad8d8c4da584d172a84"
        },
        "date": 1634149893231,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.864144143627495,
            "unit": "iter/sec",
            "range": "stddev: 0.005339015124003922",
            "extra": "mean: 127.15941896999993 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 233.56579995083118,
            "unit": "iter/sec",
            "range": "stddev: 0.00023039576055160847",
            "extra": "mean: 4.281448740399981 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 461.9091502119366,
            "unit": "iter/sec",
            "range": "stddev: 0.0001229230518881364",
            "extra": "mean: 2.1649278857999947 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16165.131643935694,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024220002008560303",
            "extra": "mean: 61.861543847998746 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 16864.301213696912,
            "unit": "iter/sec",
            "range": "stddev: 0.00000270273155055296",
            "extra": "mean: 59.296853592001526 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16592.89263063582,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037428363604225574",
            "extra": "mean: 60.26676736000081 usec\nrounds: 250"
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
          "id": "437aa845e180ad7719fe6ee988174c4e2329358e",
          "message": "Remove pin pooch",
          "timestamp": "2021-10-14T13:13:04+02:00",
          "tree_id": "85e2dcb7548d7b1c63ded7fe1d1cb172dfc09f5e",
          "url": "https://github.com/histolab/histolab/commit/437aa845e180ad7719fe6ee988174c4e2329358e"
        },
        "date": 1634210130883,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.132390244526723,
            "unit": "iter/sec",
            "range": "stddev: 0.005612826783121539",
            "extra": "mean: 140.20545226999928 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 202.33242172789866,
            "unit": "iter/sec",
            "range": "stddev: 0.0002186630653459913",
            "extra": "mean: 4.942361641599995 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 438.9877023584705,
            "unit": "iter/sec",
            "range": "stddev: 0.0001628028845519526",
            "extra": "mean: 2.2779681404000147 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15785.840687631586,
            "unit": "iter/sec",
            "range": "stddev: 0.000004849989733551044",
            "extra": "mean: 63.347909039998974 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 16678.69445157517,
            "unit": "iter/sec",
            "range": "stddev: 0.000004406165676028979",
            "extra": "mean: 59.95673119999856 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15343.56487877397,
            "unit": "iter/sec",
            "range": "stddev: 0.000005054831108320654",
            "extra": "mean: 65.17390240799796 usec\nrounds: 250"
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
          "id": "e1e9f7ac2f177a34b3e3d92d450647c89e8c1eb7",
          "message": "Clarify docstring",
          "timestamp": "2021-10-15T15:01:31+02:00",
          "tree_id": "18a8bae9023252a0b6b27b6a6fd901ddbc541c76",
          "url": "https://github.com/histolab/histolab/commit/e1e9f7ac2f177a34b3e3d92d450647c89e8c1eb7"
        },
        "date": 1634303015054,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.265176723095191,
            "unit": "iter/sec",
            "range": "stddev: 0.001688795059359425",
            "extra": "mean: 120.98954850000041 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 203.46134106739933,
            "unit": "iter/sec",
            "range": "stddev: 0.00008337846460505258",
            "extra": "mean: 4.914938605799989 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 585.2966354620146,
            "unit": "iter/sec",
            "range": "stddev: 0.00009700268309915427",
            "extra": "mean: 1.7085353637999845 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 28433.071089842,
            "unit": "iter/sec",
            "range": "stddev: 4.317541856568646e-7",
            "extra": "mean: 35.170312655999396 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29678.75905417583,
            "unit": "iter/sec",
            "range": "stddev: 8.772864859848987e-8",
            "extra": "mean: 33.69413115199973 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26221.86658975205,
            "unit": "iter/sec",
            "range": "stddev: 1.5094729020133383e-7",
            "extra": "mean: 38.136110432001715 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "767661d5c7528d1758d2fa2d2a399fc629897fff",
          "message": "add 3.9 reference in contributing files",
          "timestamp": "2021-10-15T17:03:25+02:00",
          "tree_id": "3e2ec1ff34dae3e9cfe8f30dcc0ab6087b639033",
          "url": "https://github.com/histolab/histolab/commit/767661d5c7528d1758d2fa2d2a399fc629897fff"
        },
        "date": 1634310348322,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.68745274066356,
            "unit": "iter/sec",
            "range": "stddev: 0.0011603815179435462",
            "extra": "mean: 130.0821005000003 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 156.38037445501604,
            "unit": "iter/sec",
            "range": "stddev: 0.00006269670024342498",
            "extra": "mean: 6.394664314399998 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 579.3430270396716,
            "unit": "iter/sec",
            "range": "stddev: 0.00010003556329612605",
            "extra": "mean: 1.726093097400002 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29239.982047352412,
            "unit": "iter/sec",
            "range": "stddev: 9.29209679632216e-8",
            "extra": "mean: 34.19974740000043 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 29541.053392022906,
            "unit": "iter/sec",
            "range": "stddev: 7.450655200156194e-8",
            "extra": "mean: 33.85119639200252 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26526.8506456601,
            "unit": "iter/sec",
            "range": "stddev: 9.740029440393784e-8",
            "extra": "mean: 37.697652592001305 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "pamaron",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "e5e28846fada56d04b90cf32a2772fbdb2e0786a",
          "message": "add note for data moudule",
          "timestamp": "2021-10-19T14:34:16+02:00",
          "tree_id": "d16f4e48f756478ff73e4dfbafeee63c2e52fca9",
          "url": "https://github.com/histolab/histolab/commit/e5e28846fada56d04b90cf32a2772fbdb2e0786a"
        },
        "date": 1634647004477,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.997298637862052,
            "unit": "iter/sec",
            "range": "stddev: 0.0018488402881480658",
            "extra": "mean: 142.9122939800007 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 168.29629908477938,
            "unit": "iter/sec",
            "range": "stddev: 0.00005331769629897104",
            "extra": "mean: 5.941901309999986 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 499.7895967263519,
            "unit": "iter/sec",
            "range": "stddev: 0.00011629512683157978",
            "extra": "mean: 2.0008419673999867 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 25260.95754386914,
            "unit": "iter/sec",
            "range": "stddev: 4.296088952803645e-7",
            "extra": "mean: 39.586781232000476 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 26206.045424190273,
            "unit": "iter/sec",
            "range": "stddev: 5.193060038956811e-7",
            "extra": "mean: 38.1591340399998 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 23275.273372452066,
            "unit": "iter/sec",
            "range": "stddev: 4.989900454082252e-7",
            "extra": "mean: 42.96404961600024 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "51ae30f9a2ec478165ce30df12eacc8da6886df8",
          "message": "Update numpy requirement from <1.21.3,>=1.18.4 to >=1.18.4,<1.21.4\n\nUpdates the requirements on [numpy](https://github.com/numpy/numpy) to permit the latest version.\n- [Release notes](https://github.com/numpy/numpy/releases)\n- [Changelog](https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst.txt)\n- [Commits](https://github.com/numpy/numpy/compare/v1.18.4...v1.21.3)\n\n---\nupdated-dependencies:\n- dependency-name: numpy\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-10-21T09:30:28+02:00",
          "tree_id": "16dfd3588d87e4ad6506487fa25824f7b4de2280",
          "url": "https://github.com/histolab/histolab/commit/51ae30f9a2ec478165ce30df12eacc8da6886df8"
        },
        "date": 1634801543794,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 9.66533787517807,
            "unit": "iter/sec",
            "range": "stddev: 0.005496485195738102",
            "extra": "mean: 103.46249793999846 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 314.21769643024965,
            "unit": "iter/sec",
            "range": "stddev: 0.00030659890031709474",
            "extra": "mean: 3.182506941400038 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 472.89033599384913,
            "unit": "iter/sec",
            "range": "stddev: 0.00010763100372368302",
            "extra": "mean: 2.1146551829999907 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 17641.21936988311,
            "unit": "iter/sec",
            "range": "stddev: 0.000002623570728172965",
            "extra": "mean: 56.685424008001974 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 18523.69806832174,
            "unit": "iter/sec",
            "range": "stddev: 0.0000021817042646447827",
            "extra": "mean: 53.98490065599526 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 18564.178341761195,
            "unit": "iter/sec",
            "range": "stddev: 0.000004365570280158292",
            "extra": "mean: 53.867183431999365 usec\nrounds: 250"
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
          "id": "a7d48b8b74ea5a1af4226672b112a0bcc21e16ae",
          "message": "Use Python 3.9 for benchmarks",
          "timestamp": "2021-11-03T17:19:48+01:00",
          "tree_id": "825568dda903d6b9f8339e7178fbb01a0ab84078",
          "url": "https://github.com/histolab/histolab/commit/a7d48b8b74ea5a1af4226672b112a0bcc21e16ae"
        },
        "date": 1635956517746,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.5158434507929925,
            "unit": "iter/sec",
            "range": "stddev: 0.006126181048428254",
            "extra": "mean: 133.0522657300014 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 219.8857699661005,
            "unit": "iter/sec",
            "range": "stddev: 0.00021807030952399569",
            "extra": "mean: 4.54781589619996 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 523.877703943152,
            "unit": "iter/sec",
            "range": "stddev: 0.00013356529823178672",
            "extra": "mean: 1.9088424502000065 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26737.957819782885,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032628549223882247",
            "extra": "mean: 37.400014120005835 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 20434.072765470144,
            "unit": "iter/sec",
            "range": "stddev: 0.000004558436019246481",
            "extra": "mean: 48.93787016799797 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 23577.945317460853,
            "unit": "iter/sec",
            "range": "stddev: 0.000003936032536879276",
            "extra": "mean: 42.41251671999771 usec\nrounds: 250"
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
          "id": "17d1e0e0084c77e04210053b051aa8cab68323af",
          "message": "Remove option to continue on error",
          "timestamp": "2021-11-04T15:07:27+01:00",
          "tree_id": "708e0dda95e7ea2d9041c849aeb50e0d77144bc3",
          "url": "https://github.com/histolab/histolab/commit/17d1e0e0084c77e04210053b051aa8cab68323af"
        },
        "date": 1636034999171,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.467828643119729,
            "unit": "iter/sec",
            "range": "stddev: 0.0032291451295186784",
            "extra": "mean: 133.9077324599998 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 203.4134483847342,
            "unit": "iter/sec",
            "range": "stddev: 0.00008852289519983585",
            "extra": "mean: 4.916095803600015 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 420.53812816353127,
            "unit": "iter/sec",
            "range": "stddev: 0.00010335035827939682",
            "extra": "mean: 2.3779056713999975 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16067.426884424533,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037950498638094135",
            "extra": "mean: 62.23771903199895 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 11772.43173319198,
            "unit": "iter/sec",
            "range": "stddev: 0.0000044696162521911605",
            "extra": "mean: 84.94421735999822 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 13769.895058609834,
            "unit": "iter/sec",
            "range": "stddev: 0.00000459495763333895",
            "extra": "mean: 72.62219470399918 usec\nrounds: 250"
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
          "id": "b3e1563d316c5e514c30f555c4b522fb9cae673c",
          "message": "Update numpy requirement from <1.21.4,>=1.18.4 to >=1.18.4,<1.21.5\n\nUpdates the requirements on [numpy](https://github.com/numpy/numpy) to permit the latest version.\n- [Release notes](https://github.com/numpy/numpy/releases)\n- [Changelog](https://github.com/numpy/numpy/blob/main/doc/HOWTO_RELEASE.rst.txt)\n- [Commits](https://github.com/numpy/numpy/compare/v1.18.4...v1.21.4)\n\n---\nupdated-dependencies:\n- dependency-name: numpy\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-11-05T09:49:10+01:00",
          "tree_id": "e148617fed61f372af4894f74d0226f227893408",
          "url": "https://github.com/histolab/histolab/commit/b3e1563d316c5e514c30f555c4b522fb9cae673c"
        },
        "date": 1636102270266,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.931651420336093,
            "unit": "iter/sec",
            "range": "stddev: 0.0009708768267264302",
            "extra": "mean: 126.07714925999944 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 162.63916028893075,
            "unit": "iter/sec",
            "range": "stddev: 0.00006953269270028537",
            "extra": "mean: 6.148580687599996 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 595.1149005025384,
            "unit": "iter/sec",
            "range": "stddev: 0.00009843569756349434",
            "extra": "mean: 1.6803477768000106 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29500.36018063506,
            "unit": "iter/sec",
            "range": "stddev: 8.700121599698549e-8",
            "extra": "mean: 33.8978912080006 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 22785.225386218066,
            "unit": "iter/sec",
            "range": "stddev: 1.0755192926024054e-7",
            "extra": "mean: 43.888089016001686 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26970.129681005765,
            "unit": "iter/sec",
            "range": "stddev: 9.296603936909779e-8",
            "extra": "mean: 37.07805679200237 usec\nrounds: 250"
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
          "id": "fb5368a3dccf9d2dcb0139b578b4577e12978db7",
          "message": "Move RGB->color space expectations in correct folder",
          "timestamp": "2021-11-05T11:21:48+01:00",
          "tree_id": "ee9f5e0ccec5c74ed970ccdc9085bc374dcb2cc4",
          "url": "https://github.com/histolab/histolab/commit/fb5368a3dccf9d2dcb0139b578b4577e12978db7"
        },
        "date": 1636107851468,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.534297763258088,
            "unit": "iter/sec",
            "range": "stddev: 0.0054218712075153885",
            "extra": "mean: 132.72637098000303 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 198.93498299596146,
            "unit": "iter/sec",
            "range": "stddev: 0.0002414874636074343",
            "extra": "mean: 5.026767966799992 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 453.0645073508321,
            "unit": "iter/sec",
            "range": "stddev: 0.00014379104269647992",
            "extra": "mean: 2.2071912140000105 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 14686.928336282517,
            "unit": "iter/sec",
            "range": "stddev: 0.00000428650331691163",
            "extra": "mean: 68.08775647999892 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 12357.714558539286,
            "unit": "iter/sec",
            "range": "stddev: 0.000005503343650251385",
            "extra": "mean: 80.92111168800147 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 14619.437331053154,
            "unit": "iter/sec",
            "range": "stddev: 0.000005702481753303097",
            "extra": "mean: 68.4020853440029 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "4d9e68f12930622f6e0afc16840c3828565df920",
          "message": "release: prepare v0.3.0 release",
          "timestamp": "2021-11-06T18:09:07+01:00",
          "tree_id": "369f0dabab5c017b4d8f63b92a87bc8bc1dfc1c3",
          "url": "https://github.com/histolab/histolab/commit/4d9e68f12930622f6e0afc16840c3828565df920"
        },
        "date": 1636218740984,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.690222331760172,
            "unit": "iter/sec",
            "range": "stddev: 0.00657656196854791",
            "extra": "mean: 130.03525215000067 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 218.54536999422535,
            "unit": "iter/sec",
            "range": "stddev: 0.00025864395087967564",
            "extra": "mean: 4.57570892499998 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 458.8881877140617,
            "unit": "iter/sec",
            "range": "stddev: 0.00013220705507542907",
            "extra": "mean: 2.179180085199994 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15301.657567699753,
            "unit": "iter/sec",
            "range": "stddev: 0.000005234704160308749",
            "extra": "mean: 65.35239699200292 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 12131.03022362776,
            "unit": "iter/sec",
            "range": "stddev: 0.000007151952201835764",
            "extra": "mean: 82.43322962400066 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 14798.57069517927,
            "unit": "iter/sec",
            "range": "stddev: 0.000005029856818100393",
            "extra": "mean: 67.5740935120009 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "c0ff00f719ad910b7ba5564bb1db082deeb39a40",
          "message": "fix release yml",
          "timestamp": "2021-11-06T18:52:45+01:00",
          "tree_id": "53aa9b8cb99ef08b3e43ca145b9f5de2947bdbf0",
          "url": "https://github.com/histolab/histolab/commit/c0ff00f719ad910b7ba5564bb1db082deeb39a40"
        },
        "date": 1636221422357,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.980543735717535,
            "unit": "iter/sec",
            "range": "stddev: 0.0012929811824261169",
            "extra": "mean: 125.30474528000185 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 145.98275036622263,
            "unit": "iter/sec",
            "range": "stddev: 0.00010607784246767178",
            "extra": "mean: 6.8501243981999895 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 586.3986215424549,
            "unit": "iter/sec",
            "range": "stddev: 0.00009723756233279581",
            "extra": "mean: 1.705324608999956 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 28975.468054645058,
            "unit": "iter/sec",
            "range": "stddev: 7.491801055384278e-8",
            "extra": "mean: 34.511953288005316 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 22702.889468992715,
            "unit": "iter/sec",
            "range": "stddev: 8.276808655051413e-8",
            "extra": "mean: 44.04725668799938 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26647.13046295147,
            "unit": "iter/sec",
            "range": "stddev: 2.456344336666266e-7",
            "extra": "mean: 37.52749292800354 usec\nrounds: 250"
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
          "id": "f080986c3c8fc723024ba50fa8ec8c69c27a7f9d",
          "message": "Update scipy requirement from <1.7.2,>=1.5.0 to >=1.5.0,<1.7.3\n\nUpdates the requirements on [scipy](https://github.com/scipy/scipy) to permit the latest version.\n- [Release notes](https://github.com/scipy/scipy/releases)\n- [Commits](https://github.com/scipy/scipy/compare/v1.5.0...v1.7.2)\n\n---\nupdated-dependencies:\n- dependency-name: scipy\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-11-08T09:51:09+01:00",
          "tree_id": "0705c74cf6324ddffd063b4fc7dfc012858154a8",
          "url": "https://github.com/histolab/histolab/commit/f080986c3c8fc723024ba50fa8ec8c69c27a7f9d"
        },
        "date": 1636361607020,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.82484289679311,
            "unit": "iter/sec",
            "range": "stddev: 0.003955102475006532",
            "extra": "mean: 127.79809296999872 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 189.9194765265812,
            "unit": "iter/sec",
            "range": "stddev: 0.0001651239593055172",
            "extra": "mean: 5.265389407600014 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 425.4446472530841,
            "unit": "iter/sec",
            "range": "stddev: 0.00010823221253666344",
            "extra": "mean: 2.350482034399954 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16640.23047893975,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032720902966708585",
            "extra": "mean: 60.09532147199661 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 13316.448138171594,
            "unit": "iter/sec",
            "range": "stddev: 0.000003811317531487408",
            "extra": "mean: 75.09509965600364 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15112.17781196012,
            "unit": "iter/sec",
            "range": "stddev: 0.000003033265015503201",
            "extra": "mean: 66.17179948799821 usec\nrounds: 250"
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
          "id": "94ec6cb8db6318199fef881a242fff0333c5491e",
          "message": "Add test for np_to_pil with float values [0,1] array",
          "timestamp": "2021-11-09T19:13:26+01:00",
          "tree_id": "42854ea4dfd9076a6f54c5d3a424a9ea174b2d13",
          "url": "https://github.com/histolab/histolab/commit/94ec6cb8db6318199fef881a242fff0333c5491e"
        },
        "date": 1636481740310,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.755942652452455,
            "unit": "iter/sec",
            "range": "stddev: 0.0011469596619479267",
            "extra": "mean: 128.93339272999867 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 145.8862322245633,
            "unit": "iter/sec",
            "range": "stddev: 0.0001175352218246726",
            "extra": "mean: 6.8546564316000405 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 584.5273835142051,
            "unit": "iter/sec",
            "range": "stddev: 0.0000985082716547164",
            "extra": "mean: 1.7107838369999968 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 28856.817310295155,
            "unit": "iter/sec",
            "range": "stddev: 1.1728504289437764e-7",
            "extra": "mean: 34.65385628799868 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 22658.945821455647,
            "unit": "iter/sec",
            "range": "stddev: 1.0169042623365986e-7",
            "extra": "mean: 44.13267977599844 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26576.915671202478,
            "unit": "iter/sec",
            "range": "stddev: 7.350342941759747e-8",
            "extra": "mean: 37.62663856000245 usec\nrounds: 250"
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
          "id": "21e626d6a842bf85a0c5e3b116a99fbae44ef2bf",
          "message": "Remove old note about saving tiles in /tiles directory",
          "timestamp": "2021-11-10T11:18:00+01:00",
          "tree_id": "cecae15b27fe192865e6046f4421ccdb6becbb5b",
          "url": "https://github.com/histolab/histolab/commit/21e626d6a842bf85a0c5e3b116a99fbae44ef2bf"
        },
        "date": 1636539616642,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.71456401901215,
            "unit": "iter/sec",
            "range": "stddev: 0.0009491208367364214",
            "extra": "mean: 129.62495321000006 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 150.0205229260522,
            "unit": "iter/sec",
            "range": "stddev: 0.00009769624279659913",
            "extra": "mean: 6.665754661400013 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 583.0376424890686,
            "unit": "iter/sec",
            "range": "stddev: 0.00010027363977611398",
            "extra": "mean: 1.715155123999989 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29347.480251908724,
            "unit": "iter/sec",
            "range": "stddev: 8.485097839344668e-8",
            "extra": "mean: 34.07447560800256 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 22717.317321432576,
            "unit": "iter/sec",
            "range": "stddev: 3.096943539910257e-7",
            "extra": "mean: 44.01928211200152 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27209.296402911466,
            "unit": "iter/sec",
            "range": "stddev: 9.856943610861257e-8",
            "extra": "mean: 36.7521447520046 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "1d2dff41e72c21b8d5da5a881f867bdc48581316",
          "message": "Bump sphinx from 4.2.0 to 4.3.0\n\nBumps [sphinx](https://github.com/sphinx-doc/sphinx) from 4.2.0 to 4.3.0.\n- [Release notes](https://github.com/sphinx-doc/sphinx/releases)\n- [Changelog](https://github.com/sphinx-doc/sphinx/blob/4.x/CHANGES)\n- [Commits](https://github.com/sphinx-doc/sphinx/compare/v4.2.0...v4.3.0)\n\n---\nupdated-dependencies:\n- dependency-name: sphinx\n  dependency-type: direct:production\n  update-type: version-update:semver-minor\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-11-11T10:48:38+01:00",
          "tree_id": "30a5a5d4ea239dbc699c561edb7429e7ed30803f",
          "url": "https://github.com/histolab/histolab/commit/1d2dff41e72c21b8d5da5a881f867bdc48581316"
        },
        "date": 1636624256372,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.670623321068092,
            "unit": "iter/sec",
            "range": "stddev: 0.00783747025132979",
            "extra": "mean: 130.3675018500003 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 209.84268853816351,
            "unit": "iter/sec",
            "range": "stddev: 0.0004044689380684808",
            "extra": "mean: 4.765474589400014 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 447.43840307053097,
            "unit": "iter/sec",
            "range": "stddev: 0.00015692073031325465",
            "extra": "mean: 2.2349445044000102 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 14794.572471333397,
            "unit": "iter/sec",
            "range": "stddev: 0.000005656477982868394",
            "extra": "mean: 67.5923553680002 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 12838.627138451739,
            "unit": "iter/sec",
            "range": "stddev: 0.000006479680649057928",
            "extra": "mean: 77.88994798400176 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15085.830578427393,
            "unit": "iter/sec",
            "range": "stddev: 0.000006113976013943835",
            "extra": "mean: 66.2873677919988 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "fb1370dcca89a8666067109fb17ab74a3d103783",
          "message": "Add 3.10 references where needed",
          "timestamp": "2021-11-11T12:44:22+01:00",
          "tree_id": "640f2c2b98d970c21cb9feea54e67a21ae01e9d0",
          "url": "https://github.com/histolab/histolab/commit/fb1370dcca89a8666067109fb17ab74a3d103783"
        },
        "date": 1636631202016,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.746229942575961,
            "unit": "iter/sec",
            "range": "stddev: 0.001160599303914334",
            "extra": "mean: 129.0950575200013 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 148.01117883568543,
            "unit": "iter/sec",
            "range": "stddev: 0.00009383400033896255",
            "extra": "mean: 6.756246439400025 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 584.4833950177022,
            "unit": "iter/sec",
            "range": "stddev: 0.00009952534666018621",
            "extra": "mean: 1.710912591399989 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29423.28894276937,
            "unit": "iter/sec",
            "range": "stddev: 1.4158205821771772e-7",
            "extra": "mean: 33.98668320000115 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 23036.141127409373,
            "unit": "iter/sec",
            "range": "stddev: 1.1323491881440623e-7",
            "extra": "mean: 43.410048343997914 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27262.71264140192,
            "unit": "iter/sec",
            "range": "stddev: 1.3312542040675738e-7",
            "extra": "mean: 36.68013572799691 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "7cea37b5efc09991eccd5bac63c5427e69f0dccc",
          "message": "release: prepare v0.3.1 release",
          "timestamp": "2021-11-11T13:06:36+01:00",
          "tree_id": "d14a9fbabddfd1d7ac9a5fad80d7003d3b7100e2",
          "url": "https://github.com/histolab/histolab/commit/7cea37b5efc09991eccd5bac63c5427e69f0dccc"
        },
        "date": 1636632535359,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.702321126289749,
            "unit": "iter/sec",
            "range": "stddev: 0.0008100564512312342",
            "extra": "mean: 129.8309929699991 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 149.02664774041807,
            "unit": "iter/sec",
            "range": "stddev: 0.0000651010469907658",
            "extra": "mean: 6.710209315999975 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 591.5063675449205,
            "unit": "iter/sec",
            "range": "stddev: 0.00009931035821900857",
            "extra": "mean: 1.6905988758000263 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29237.910442316643,
            "unit": "iter/sec",
            "range": "stddev: 8.37091654064245e-8",
            "extra": "mean: 34.20217056799925 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 22498.714860807777,
            "unit": "iter/sec",
            "range": "stddev: 9.4734352954298e-8",
            "extra": "mean: 44.4469831360002 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26724.90034710159,
            "unit": "iter/sec",
            "range": "stddev: 9.908391264283856e-8",
            "extra": "mean: 37.41828732799945 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mtageld@emory.edu",
            "name": "kheffah",
            "username": "kheffah"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "29ebb9a28abb72b98b4daad7c59fb1649700f6b4",
          "message": "remove underscore from list comprehension in Slide class",
          "timestamp": "2021-11-15T12:21:07+01:00",
          "tree_id": "ba23b31d0f00e95849089816d0518f3d8b708c97",
          "url": "https://github.com/histolab/histolab/commit/29ebb9a28abb72b98b4daad7c59fb1649700f6b4"
        },
        "date": 1636975399855,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.967458565869303,
            "unit": "iter/sec",
            "range": "stddev: 0.0017617953969468597",
            "extra": "mean: 143.52435548000045 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 150.44247183151361,
            "unit": "iter/sec",
            "range": "stddev: 0.00007898813698432419",
            "extra": "mean: 6.64705909060002 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 505.3801581976831,
            "unit": "iter/sec",
            "range": "stddev: 0.00010586207082677048",
            "extra": "mean: 1.9787084708000011 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26474.510204119513,
            "unit": "iter/sec",
            "range": "stddev: 6.907129415307413e-7",
            "extra": "mean: 37.77218132800044 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 20642.377118945707,
            "unit": "iter/sec",
            "range": "stddev: 8.61595053599011e-7",
            "extra": "mean: 48.44403307999801 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24646.23651277712,
            "unit": "iter/sec",
            "range": "stddev: 8.294620790572559e-7",
            "extra": "mean: 40.57414605599433 usec\nrounds: 250"
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
          "id": "97abd621b133e1f310355c95dc3e8ea7ddc62bc9",
          "message": "Update expectation after updating to skimage 0.18.3",
          "timestamp": "2021-11-15T14:53:31+01:00",
          "tree_id": "709c136817b2eccd3f344c656b207bf2b7b89a9d",
          "url": "https://github.com/histolab/histolab/commit/97abd621b133e1f310355c95dc3e8ea7ddc62bc9"
        },
        "date": 1636984551714,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.886252991483174,
            "unit": "iter/sec",
            "range": "stddev: 0.006214839546127328",
            "extra": "mean: 126.8029317700001 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 213.04220823568295,
            "unit": "iter/sec",
            "range": "stddev: 0.00040374480718794695",
            "extra": "mean: 4.693905532999953 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 437.0800900876756,
            "unit": "iter/sec",
            "range": "stddev: 0.00015328885900347264",
            "extra": "mean: 2.287910208400035 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15270.970664721039,
            "unit": "iter/sec",
            "range": "stddev: 0.0000051361156739649",
            "extra": "mean: 65.48372215200425 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 13401.380445708688,
            "unit": "iter/sec",
            "range": "stddev: 0.0000053771101492025984",
            "extra": "mean: 74.61917852800116 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15389.143493773505,
            "unit": "iter/sec",
            "range": "stddev: 0.0000056027260047222705",
            "extra": "mean: 64.9808743679987 usec\nrounds: 250"
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
          "id": "7463921b65c4be69381b5b56e426b21296de792f",
          "message": "Fix error message",
          "timestamp": "2021-11-16T17:51:16+01:00",
          "tree_id": "5d79c760ca93c1ec0f148d9ada2430d53e6b4448",
          "url": "https://github.com/histolab/histolab/commit/7463921b65c4be69381b5b56e426b21296de792f"
        },
        "date": 1637081616056,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.865394070928437,
            "unit": "iter/sec",
            "range": "stddev: 0.0064337659590781105",
            "extra": "mean: 127.1392114599999 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 211.36474106020677,
            "unit": "iter/sec",
            "range": "stddev: 0.00028635309098205695",
            "extra": "mean: 4.731158068199995 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 464.480396117877,
            "unit": "iter/sec",
            "range": "stddev: 0.00012648031297175352",
            "extra": "mean: 2.152943392999987 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15832.37996388067,
            "unit": "iter/sec",
            "range": "stddev: 0.00000495480977399381",
            "extra": "mean: 63.16169788000025 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 12942.25509481545,
            "unit": "iter/sec",
            "range": "stddev: 0.000006076705997374077",
            "extra": "mean: 77.26628726400168 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15571.231149109828,
            "unit": "iter/sec",
            "range": "stddev: 0.000004616266880761596",
            "extra": "mean: 64.22099771199964 usec\nrounds: 250"
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
          "id": "7f57e63e43e2fe4c2dcbd6988556aa42b076a1c6",
          "message": "Fix link to installation instructions",
          "timestamp": "2021-11-16T18:57:08+01:00",
          "tree_id": "8d3f0c2c523c520b0461aa64c13576cbc4074b5d",
          "url": "https://github.com/histolab/histolab/commit/7f57e63e43e2fe4c2dcbd6988556aa42b076a1c6"
        },
        "date": 1637085557666,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.916390574200227,
            "unit": "iter/sec",
            "range": "stddev: 0.005362039726785859",
            "extra": "mean: 126.32019486999951 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 244.4341593698528,
            "unit": "iter/sec",
            "range": "stddev: 0.00028883060417514905",
            "extra": "mean: 4.091081224399992 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 470.8467355583675,
            "unit": "iter/sec",
            "range": "stddev: 0.00011126060343882131",
            "extra": "mean: 2.1238333506000004 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15754.700177311444,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023821940533187653",
            "extra": "mean: 63.47312159200043 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 14207.354112837897,
            "unit": "iter/sec",
            "range": "stddev: 0.000003699069827106245",
            "extra": "mean: 70.3860825920001 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16132.788633223541,
            "unit": "iter/sec",
            "range": "stddev: 0.000003831797127203772",
            "extra": "mean: 61.98556385599819 usec\nrounds: 250"
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
          "id": "943dbb7adbe2f5ff765c10f3d8bf0096416b31e5",
          "message": "Fix SlideSet not passing along args to Slide",
          "timestamp": "2021-11-18T18:16:43+01:00",
          "tree_id": "f6624c3fd74698d4c5bd4e0524073a365bee8e77",
          "url": "https://github.com/histolab/histolab/commit/943dbb7adbe2f5ff765c10f3d8bf0096416b31e5"
        },
        "date": 1637255961423,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.77589471441042,
            "unit": "iter/sec",
            "range": "stddev: 0.0011851235609872347",
            "extra": "mean: 128.6025643000005 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 158.50341312169274,
            "unit": "iter/sec",
            "range": "stddev: 0.00013220783226055262",
            "extra": "mean: 6.30901240739995 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 587.6612009076766,
            "unit": "iter/sec",
            "range": "stddev: 0.00009711624436674655",
            "extra": "mean: 1.701660750200017 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29345.525006350308,
            "unit": "iter/sec",
            "range": "stddev: 7.730086028242672e-8",
            "extra": "mean: 34.076745936002226 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 23059.712829916854,
            "unit": "iter/sec",
            "range": "stddev: 8.496485761883312e-8",
            "extra": "mean: 43.36567447200105 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27232.800184967866,
            "unit": "iter/sec",
            "range": "stddev: 8.94155253700047e-8",
            "extra": "mean: 36.72042511999871 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "7f455d36183e81892dc58a1eb37a2c318430591f",
          "message": "release: prepare v0.4.0 release",
          "timestamp": "2021-11-18T18:51:41+01:00",
          "tree_id": "bf3e6796c89248135f138ce10544dfc1c1d49bb2",
          "url": "https://github.com/histolab/histolab/commit/7f455d36183e81892dc58a1eb37a2c318430591f"
        },
        "date": 1637258129035,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 6.96026783662434,
            "unit": "iter/sec",
            "range": "stddev: 0.0021914573253125715",
            "extra": "mean: 143.67263207000235 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 153.0792390155327,
            "unit": "iter/sec",
            "range": "stddev: 0.00008557678559176564",
            "extra": "mean: 6.532564483800002 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 504.9898975912614,
            "unit": "iter/sec",
            "range": "stddev: 0.00011789070817219834",
            "extra": "mean: 1.9802376339999568 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26019.90318056246,
            "unit": "iter/sec",
            "range": "stddev: 9.739061844965745e-8",
            "extra": "mean: 38.43211840799722 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 20175.271564504972,
            "unit": "iter/sec",
            "range": "stddev: 1.0648929148981354e-7",
            "extra": "mean: 49.56562774398208 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24113.241453299386,
            "unit": "iter/sec",
            "range": "stddev: 1.2093957151580662e-7",
            "extra": "mean: 41.470990199999505 usec\nrounds: 250"
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
          "id": "641f2aca65a199604d53f498846e2aeb67f3bf71",
          "message": "Try tp use http instead of ftp",
          "timestamp": "2021-11-19T16:17:07+01:00",
          "tree_id": "3f60cdb5b057421490c845b92b21de25805b01c5",
          "url": "https://github.com/histolab/histolab/commit/641f2aca65a199604d53f498846e2aeb67f3bf71"
        },
        "date": 1637335168224,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.249129810470523,
            "unit": "iter/sec",
            "range": "stddev: 0.005321577651787697",
            "extra": "mean: 137.94759180000014 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 150.30104864320455,
            "unit": "iter/sec",
            "range": "stddev: 0.00013316690692187456",
            "extra": "mean: 6.653313526600016 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 488.97317249050565,
            "unit": "iter/sec",
            "range": "stddev: 0.00012656592199298102",
            "extra": "mean: 2.045101973400017 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 25996.054428250005,
            "unit": "iter/sec",
            "range": "stddev: 6.880465215996197e-7",
            "extra": "mean: 38.46737599200037 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 19906.63223813488,
            "unit": "iter/sec",
            "range": "stddev: 4.202351606909732e-7",
            "extra": "mean: 50.234514207999155 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 23551.42573563741,
            "unit": "iter/sec",
            "range": "stddev: 4.40885026675308e-7",
            "extra": "mean: 42.46027443199864 usec\nrounds: 250"
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
          "id": "9231c477a5c098e7490ee4fee59f0b5873c0fe15",
          "message": "None as default level for Tile, instead of 0",
          "timestamp": "2021-11-23T13:57:21+01:00",
          "tree_id": "300cba9bf01fa57f97c4e18d71a6e7bf9034562a",
          "url": "https://github.com/histolab/histolab/commit/9231c477a5c098e7490ee4fee59f0b5873c0fe15"
        },
        "date": 1637672386302,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.995556672786093,
            "unit": "iter/sec",
            "range": "stddev: 0.0014263755476542658",
            "extra": "mean: 125.06946556999948 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 147.1427463861575,
            "unit": "iter/sec",
            "range": "stddev: 0.00011930738870747532",
            "extra": "mean: 6.796121620400007 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 577.7033545808227,
            "unit": "iter/sec",
            "range": "stddev: 0.00009849600990778485",
            "extra": "mean: 1.7309921988000099 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 28981.322746592075,
            "unit": "iter/sec",
            "range": "stddev: 7.516302774305298e-8",
            "extra": "mean: 34.50498131999825 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 22421.035732274246,
            "unit": "iter/sec",
            "range": "stddev: 8.597082848566112e-8",
            "extra": "mean: 44.60097258399787 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26691.794774961676,
            "unit": "iter/sec",
            "range": "stddev: 9.983239719233041e-8",
            "extra": "mean: 37.4646968639986 usec\nrounds: 250"
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
          "id": "348f9451780bae10fdfcd85e05b05455106c896e",
          "message": "Update scipy requirement from <1.7.3,>=1.5.0 to >=1.5.0,<1.7.4\n\nUpdates the requirements on [scipy](https://github.com/scipy/scipy) to permit the latest version.\n- [Release notes](https://github.com/scipy/scipy/releases)\n- [Commits](https://github.com/scipy/scipy/compare/v1.5.0...v1.7.3)\n\n---\nupdated-dependencies:\n- dependency-name: scipy\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-11-25T07:49:18+01:00",
          "tree_id": "771c6c89abbd8e8efbfa1a74e7d0048626625469",
          "url": "https://github.com/histolab/histolab/commit/348f9451780bae10fdfcd85e05b05455106c896e"
        },
        "date": 1637823162640,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.070397626256177,
            "unit": "iter/sec",
            "range": "stddev: 0.0027866993494796996",
            "extra": "mean: 141.43476122000038 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 155.82401921369382,
            "unit": "iter/sec",
            "range": "stddev: 0.00010736878027224966",
            "extra": "mean: 6.417495871600004 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 524.7608174461003,
            "unit": "iter/sec",
            "range": "stddev: 0.00010538307223591152",
            "extra": "mean: 1.905630082799985 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26928.48968621725,
            "unit": "iter/sec",
            "range": "stddev: 9.448646903417058e-7",
            "extra": "mean: 37.13539124000067 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 20810.92515926263,
            "unit": "iter/sec",
            "range": "stddev: 9.713984736151121e-7",
            "extra": "mean: 48.051684024 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24864.032638759094,
            "unit": "iter/sec",
            "range": "stddev: 8.032227994441992e-7",
            "extra": "mean: 40.2187374239993 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "patrick.arminio@gmail.com",
            "name": "Patrick Arminio",
            "username": "patrick91"
          },
          "committer": {
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "0c49ba5e21925a3271de221fcf0f28fbbf266e55",
          "message": "Fix yml",
          "timestamp": "2021-11-26T18:18:21+01:00",
          "tree_id": "5302a50b29093bf9a767c8d37949c88c88a7962d",
          "url": "https://github.com/histolab/histolab/commit/0c49ba5e21925a3271de221fcf0f28fbbf266e55"
        },
        "date": 1637947270561,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.056057384575455,
            "unit": "iter/sec",
            "range": "stddev: 0.007856639910836303",
            "extra": "mean: 141.7222034200006 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 196.96192972884668,
            "unit": "iter/sec",
            "range": "stddev: 0.00022364937729738",
            "extra": "mean: 5.077123286599999 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 443.0261670548064,
            "unit": "iter/sec",
            "range": "stddev: 0.0001376175289726909",
            "extra": "mean: 2.2572030149999933 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15938.63215327765,
            "unit": "iter/sec",
            "range": "stddev: 0.000004552714123756143",
            "extra": "mean: 62.74064112800034 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 13611.456628802023,
            "unit": "iter/sec",
            "range": "stddev: 0.0000064569960386418905",
            "extra": "mean: 73.46752278400436 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 15365.407940103201,
            "unit": "iter/sec",
            "range": "stddev: 0.0000051091404931428015",
            "extra": "mean: 65.08125289599593 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "b8eae2b86aeccf8b822aff6f58f508fae3faad3d",
          "message": "Bump sphinx from 4.3.0 to 4.3.1\n\nBumps [sphinx](https://github.com/sphinx-doc/sphinx) from 4.3.0 to 4.3.1.\n- [Release notes](https://github.com/sphinx-doc/sphinx/releases)\n- [Changelog](https://github.com/sphinx-doc/sphinx/blob/4.x/CHANGES)\n- [Commits](https://github.com/sphinx-doc/sphinx/compare/v4.3.0...v4.3.1)\n\n---\nupdated-dependencies:\n- dependency-name: sphinx\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-11-29T10:01:15+01:00",
          "tree_id": "25b37535100f92e398212284e4a81dab18abb605",
          "url": "https://github.com/histolab/histolab/commit/b8eae2b86aeccf8b822aff6f58f508fae3faad3d"
        },
        "date": 1638176621211,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.153154146571599,
            "unit": "iter/sec",
            "range": "stddev: 0.0025134510705007037",
            "extra": "mean: 139.79846925000004 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 154.50822622125068,
            "unit": "iter/sec",
            "range": "stddev: 0.00007186645531975937",
            "extra": "mean: 6.472147305399995 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 518.4823120153309,
            "unit": "iter/sec",
            "range": "stddev: 0.00010910834065415686",
            "extra": "mean: 1.9287061039999969 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 26667.227705316684,
            "unit": "iter/sec",
            "range": "stddev: 5.811806121072945e-7",
            "extra": "mean: 37.499211055996966 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 21065.44197534674,
            "unit": "iter/sec",
            "range": "stddev: 9.174756845639028e-7",
            "extra": "mean: 47.471114120003676 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 24645.672741546972,
            "unit": "iter/sec",
            "range": "stddev: 7.588941122455062e-7",
            "extra": "mean: 40.57507419199919 usec\nrounds: 250"
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
          "id": "cf25abb32b5110dc5dad7f486880f5ffc7fe499f",
          "message": "BICUBIC for upsampling and LANCZOS for downsampling",
          "timestamp": "2021-12-02T20:54:43+01:00",
          "tree_id": "82d36dc6c329139b825ffaab0e7f93f3472a2041",
          "url": "https://github.com/histolab/histolab/commit/cf25abb32b5110dc5dad7f486880f5ffc7fe499f"
        },
        "date": 1638475024253,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.699532869100068,
            "unit": "iter/sec",
            "range": "stddev: 0.005596218321719151",
            "extra": "mean: 114.94870070000047 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 266.9001716055354,
            "unit": "iter/sec",
            "range": "stddev: 0.00024407051796516608",
            "extra": "mean: 3.746719209600016 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 468.69598595131424,
            "unit": "iter/sec",
            "range": "stddev: 0.00012979195761607783",
            "extra": "mean: 2.133579185600013 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 16513.138622600436,
            "unit": "iter/sec",
            "range": "stddev: 0.00000303824236296388",
            "extra": "mean: 60.55783959999985 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 14411.844731498815,
            "unit": "iter/sec",
            "range": "stddev: 0.000004840888769575557",
            "extra": "mean: 69.38736980800104 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16995.34663425438,
            "unit": "iter/sec",
            "range": "stddev: 0.000004391454750007401",
            "extra": "mean: 58.83963543200025 usec\nrounds: 250"
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
          "id": "19722a070f97ecd5a72a6bae95b4e8f6a3937c6c",
          "message": "Remove deprecated indices argument and adapt to new output",
          "timestamp": "2021-12-06T16:23:18+01:00",
          "tree_id": "5a980401ad7b000fcec1035afadc87662b690628",
          "url": "https://github.com/histolab/histolab/commit/19722a070f97ecd5a72a6bae95b4e8f6a3937c6c"
        },
        "date": 1638804336418,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.241959544600908,
            "unit": "iter/sec",
            "range": "stddev: 0.001431328334118123",
            "extra": "mean: 121.3303698699994 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 176.97234258617715,
            "unit": "iter/sec",
            "range": "stddev: 0.00008194309056533478",
            "extra": "mean: 5.650600457600021 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 574.4080411681293,
            "unit": "iter/sec",
            "range": "stddev: 0.00009732429414256991",
            "extra": "mean: 1.7409227036000006 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29338.376193688186,
            "unit": "iter/sec",
            "range": "stddev: 7.39188474921293e-7",
            "extra": "mean: 34.08504933600034 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 23157.29615925221,
            "unit": "iter/sec",
            "range": "stddev: 1.6293537495090074e-7",
            "extra": "mean: 43.182934359997034 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26819.863684814576,
            "unit": "iter/sec",
            "range": "stddev: 8.839196744012409e-7",
            "extra": "mean: 37.285797263995846 usec\nrounds: 250"
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
          "id": "192f08132bdadb6f7e28503924907e79042dadae",
          "message": "Use np.pad for conciseness",
          "timestamp": "2021-12-07T14:20:33+01:00",
          "tree_id": "fcf47077e03ba6e53719008e4edf04d52a1ebcec",
          "url": "https://github.com/histolab/histolab/commit/192f08132bdadb6f7e28503924907e79042dadae"
        },
        "date": 1638883393627,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.310919038441304,
            "unit": "iter/sec",
            "range": "stddev: 0.004802338851270709",
            "extra": "mean: 136.78170894000232 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 163.81234380738417,
            "unit": "iter/sec",
            "range": "stddev: 0.00010274488215098392",
            "extra": "mean: 6.104546072399967 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 531.4410244467995,
            "unit": "iter/sec",
            "range": "stddev: 0.00012491374736387629",
            "extra": "mean: 1.881676336599992 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 27530.85115714995,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014991961544604994",
            "extra": "mean: 36.32288716000312 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 21422.930825974843,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016298180891453904",
            "extra": "mean: 46.67895387999488 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 25422.778161953043,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015307393392286183",
            "extra": "mean: 39.33480415199347 usec\nrounds: 250"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "committer": {
            "email": "ernesto.arbitrio@gmail.com",
            "name": "Arbitrio",
            "username": "ernestoarbitrio"
          },
          "distinct": true,
          "id": "0f74c878d77e68595243143b3698657265ecf1af",
          "message": "release: prepare v0.4.1 release",
          "timestamp": "2021-12-07T15:54:52+01:00",
          "tree_id": "3366dbf311ae81a340986f588bcfa7f559fbeee6",
          "url": "https://github.com/histolab/histolab/commit/0f74c878d77e68595243143b3698657265ecf1af"
        },
        "date": 1638889042699,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.9537375449775904,
            "unit": "iter/sec",
            "range": "stddev: 0.0015905948149467835",
            "extra": "mean: 125.72705527999886 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 156.79986844425784,
            "unit": "iter/sec",
            "range": "stddev: 0.00010918885246434773",
            "extra": "mean: 6.377556371200012 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 581.6385269005054,
            "unit": "iter/sec",
            "range": "stddev: 0.00009994531473384835",
            "extra": "mean: 1.7192808828000126 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29999.386872526145,
            "unit": "iter/sec",
            "range": "stddev: 8.940182211070398e-8",
            "extra": "mean: 33.3340146000055 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 23203.772222324103,
            "unit": "iter/sec",
            "range": "stddev: 8.140775869730046e-8",
            "extra": "mean: 43.09644097600267 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27257.337710473963,
            "unit": "iter/sec",
            "range": "stddev: 1.036436181789394e-7",
            "extra": "mean: 36.68736875999957 usec\nrounds: 250"
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
          "id": "6578345f511ec277c7b2a33456fc89d4962069a8",
          "message": "Update docs",
          "timestamp": "2021-12-13T18:28:30+01:00",
          "tree_id": "134525cb58efa9dfe97d9f5538547913eed786aa",
          "url": "https://github.com/histolab/histolab/commit/6578345f511ec277c7b2a33456fc89d4962069a8"
        },
        "date": 1639416712094,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.720041559169977,
            "unit": "iter/sec",
            "range": "stddev: 0.001462025634368152",
            "extra": "mean: 129.53298144000087 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 144.95360179275963,
            "unit": "iter/sec",
            "range": "stddev: 0.00008844886733093272",
            "extra": "mean: 6.898759241800019 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 583.0814224701049,
            "unit": "iter/sec",
            "range": "stddev: 0.00009923518045517138",
            "extra": "mean: 1.715026343600016 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29222.742880554833,
            "unit": "iter/sec",
            "range": "stddev: 9.055447850360917e-8",
            "extra": "mean: 34.21992261600508 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 22983.765744354667,
            "unit": "iter/sec",
            "range": "stddev: 1.0161319260547464e-7",
            "extra": "mean: 43.508971120001206 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 26660.28103596661,
            "unit": "iter/sec",
            "range": "stddev: 9.083175792919281e-8",
            "extra": "mean: 37.508981943998606 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "2a5233d40309b97c1a1d7c318f772b55589fd05f",
          "message": "Update scikit-image requirement\n\nUpdates the requirements on [scikit-image](https://github.com/scikit-image/scikit-image) to permit the latest version.\n- [Release notes](https://github.com/scikit-image/scikit-image/releases)\n- [Changelog](https://github.com/scikit-image/scikit-image/blob/main/RELEASE.txt)\n- [Commits](https://github.com/scikit-image/scikit-image/compare/v0.19.0...v0.19.1)\n\n---\nupdated-dependencies:\n- dependency-name: scikit-image\n  dependency-type: direct:production\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-12-16T10:26:50+01:00",
          "tree_id": "304cc7ab2aaf141526cab66336807080f6dbdcae",
          "url": "https://github.com/histolab/histolab/commit/2a5233d40309b97c1a1d7c318f772b55589fd05f"
        },
        "date": 1639646953406,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.078935194958389,
            "unit": "iter/sec",
            "range": "stddev: 0.0011290618878184125",
            "extra": "mean: 123.77868813999697 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 169.89716120245367,
            "unit": "iter/sec",
            "range": "stddev: 0.00003680238236312777",
            "extra": "mean: 5.885913530999937 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 514.7432013182945,
            "unit": "iter/sec",
            "range": "stddev: 0.00009623073902495713",
            "extra": "mean: 1.9427162854000357 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 29740.150464969876,
            "unit": "iter/sec",
            "range": "stddev: 8.116530115954132e-8",
            "extra": "mean: 33.62457769599632 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 23060.705608034128,
            "unit": "iter/sec",
            "range": "stddev: 9.299840451264974e-8",
            "extra": "mean: 43.363807551994796 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 27024.897476795126,
            "unit": "iter/sec",
            "range": "stddev: 1.3309939315008618e-7",
            "extra": "mean: 37.00291558399613 usec\nrounds: 250"
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
          "id": "8758cd8985626316cd69e4c6033d0b20266a4695",
          "message": "Specify large_image version",
          "timestamp": "2021-12-20T12:37:21+01:00",
          "tree_id": "c1e6120f144277d844b99eaac8e94e9cb4f7424a",
          "url": "https://github.com/histolab/histolab/commit/8758cd8985626316cd69e4c6033d0b20266a4695"
        },
        "date": 1640000366013,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 8.785912111984834,
            "unit": "iter/sec",
            "range": "stddev: 0.008817527978052605",
            "extra": "mean: 113.81857537999991 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 273.1207781227042,
            "unit": "iter/sec",
            "range": "stddev: 0.0005449958312223811",
            "extra": "mean: 3.661383827600011 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 475.7405715488757,
            "unit": "iter/sec",
            "range": "stddev: 0.00012969317389922725",
            "extra": "mean: 2.101985955799995 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 15374.611746359264,
            "unit": "iter/sec",
            "range": "stddev: 0.000003954554057157636",
            "extra": "mean: 65.04229287200064 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 13959.168810548821,
            "unit": "iter/sec",
            "range": "stddev: 0.0000053985434571794275",
            "extra": "mean: 71.6375031759992 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 16466.308361087926,
            "unit": "iter/sec",
            "range": "stddev: 0.000004286542533586409",
            "extra": "mean: 60.73006639199912 usec\nrounds: 250"
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
            "email": "98marcolini@gmail.com",
            "name": "Alessia Marcolini",
            "username": "alessiamarcolini"
          },
          "distinct": true,
          "id": "7ec27645d25e90e7ca63eab4a9b357ab70a411f1",
          "message": "Bump sphinx from 4.3.1 to 4.3.2\n\nBumps [sphinx](https://github.com/sphinx-doc/sphinx) from 4.3.1 to 4.3.2.\n- [Release notes](https://github.com/sphinx-doc/sphinx/releases)\n- [Changelog](https://github.com/sphinx-doc/sphinx/blob/4.x/CHANGES)\n- [Commits](https://github.com/sphinx-doc/sphinx/compare/v4.3.1...v4.3.2)\n\n---\nupdated-dependencies:\n- dependency-name: sphinx\n  dependency-type: direct:production\n  update-type: version-update:semver-patch\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",
          "timestamp": "2021-12-20T13:15:01+01:00",
          "tree_id": "b0ee9bbe60acfc97093381478072e13599737622",
          "url": "https://github.com/histolab/histolab/commit/7ec27645d25e90e7ca63eab4a9b357ab70a411f1"
        },
        "date": 1640002617992,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_difference",
            "value": 7.959877785176321,
            "unit": "iter/sec",
            "range": "stddev: 0.0058738144026342105",
            "extra": "mean: 125.63006958000031 msec\nrounds: 100"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksFilterUtil::test_mask_percent",
            "value": 166.0523341376163,
            "unit": "iter/sec",
            "range": "stddev: 0.00009166517951112619",
            "extra": "mean: 6.022197792000005 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_apply_mask_image",
            "value": 622.8461383382568,
            "unit": "iter/sec",
            "range": "stddev: 0.00008949232721808549",
            "extra": "mean: 1.6055329533999896 msec\nrounds: 50"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image0]",
            "value": 33908.37099272302,
            "unit": "iter/sec",
            "range": "stddev: 8.967622012296808e-8",
            "extra": "mean: 29.49124274399992 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image1]",
            "value": 26262.59729758014,
            "unit": "iter/sec",
            "range": "stddev: 1.063972894315167e-7",
            "extra": "mean: 38.076965072001485 usec\nrounds: 250"
          },
          {
            "name": "tests/benchmarks/test_benchmarks.py::TestDescribeBenchmarksUtil::test_np_to_pil[image2]",
            "value": 30920.095276105123,
            "unit": "iter/sec",
            "range": "stddev: 1.1636110578646075e-7",
            "extra": "mean: 32.34142686399787 usec\nrounds: 250"
          }
        ]
      }
    ]
  }
}