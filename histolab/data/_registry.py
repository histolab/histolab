# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2022 All Histolab Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

# flake8: noqa

# in legacy datasets we need to put our sample data within the data dir
legacy_datasets = ["cmu_small_region.svs"]

# Registry of datafiles that can be downloaded along with their SHA256 hashes
# To generate the SHA256 hash, use the command
# openssl sha256 filename
registry = {
    "histolab/broken.svs": "b1325916876afa17ad5e02d2e7298ee883e758ed25369470d85bc0990e928e11",
    "histolab/kidney.png": "5c6dc1b9ae10a2865302d9c8eda360362ec47732cb3e9766c38ed90cb9f4c371",
    "data/cmu_small_region.svs": "ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7",
    "aperio/JP2K-33003-1.svs": "6205ccf75a8fa6c32df7c5c04b7377398971a490fb6b320d50d91f7ba6a0e6fd",
    "aperio/JP2K-33003-2.svs": "1a13cef86b55b51127cebd94a1f6069f7de494c98e3e708640d1ce7181d9e3fd",
    "tcga/breast/TCGA-A8-A082-01A-01-TS1.3cad4a77-47a6-4658-becf-d8cffa161d3a.svs": "e955f47b83c8a5ae382ff8559493548f90f85c17c86315dd03134c041f44df70",
    "tcga/breast/TCGA-A1-A0SH-01Z-00-DX1.90E71B08-E1D9-4FC2-85AC-062E56DDF17C.svs": "6de90fe92400e592839ab7f87c15d9924bc539c61ee3b3bc8ef044f98d16031b",
    "tcga/breast/TCGA-E9-A24A-01Z-00-DX1.F0342837-5750-4172-B60D-5F902E2A02FD.svs": "55c694262c4d44b342e08eb3ef2082eeb9e9deeb3cb445e4776419bb9fa7dc21",
    "tcga/breast/TCGA-BH-A201-01Z-00-DX1.6D6E3224-50A0-45A2-B231-EEF27CA7EFD2.svs": "e1ccf3360078844abbec4b96c5da59a029a441c1ab6d7f694ec80d9d79bd3837",
    "tcga/prostate/TCGA-CH-5753-01A-01-BS1.4311c533-f9c1-4c6f-8b10-922daa3c2e3e.svs": "93ed7aa906c9e127c8241bc5da197902ebb71ccda4db280aefbe0ecd952b9089",
    "tcga/ovarian/TCGA-13-1404-01A-01-TS1.cecf7044-1d29-4d14-b137-821f8d48881e.svs": "6796e23af7cd219b9ff2274c087759912529fec9f49e2772a868ba9d85d389d6",
    "9798433/?format=tif": "7db49ff9fc3f6022ae334cf019e94ef4450f7d4cf0d71783e0f6ea82965d3a52",
    "9798554/?format=tif": "8a4318ac713b4cf50c3314760da41ab7653e10e90531ecd0c787f1386857a4ef",
}

APERIO_REPO_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio"
TCGA_REPO_URL = "https://api.gdc.cancer.gov/data"
IDR_REPO_URL = "https://idr.openmicroscopy.org/webclient/render_image_download"

registry_urls = {
    "histolab/broken.svs": "https://raw.githubusercontent.com/histolab/histolab/master/tests/fixtures/svs-images/broken.svs",
    "histolab/kidney.png": "https://user-images.githubusercontent.com/4196091/100275351-132cc880-2f60-11eb-8cc8-7a3bf3723260.png",
    "aperio/JP2K-33003-1.svs": f"{APERIO_REPO_URL}/JP2K-33003-1.svs",
    "aperio/JP2K-33003-2.svs": f"{APERIO_REPO_URL}/JP2K-33003-2.svs",
    "tcga/breast/TCGA-A8-A082-01A-01-TS1.3cad4a77-47a6-4658-becf-d8cffa161d3a.svs": f"{TCGA_REPO_URL}/ad9ed74a-2725-49e6-bf7a-ef100e299989",
    "tcga/breast/TCGA-A1-A0SH-01Z-00-DX1.90E71B08-E1D9-4FC2-85AC-062E56DDF17C.svs": f"{TCGA_REPO_URL}/3845b8bd-cbe0-49cf-a418-a8120f6c23db",
    "tcga/breast/TCGA-E9-A24A-01Z-00-DX1.F0342837-5750-4172-B60D-5F902E2A02FD.svs": f"{TCGA_REPO_URL}/682e4d74-2200-4f34-9e96-8dee968b1568",
    "tcga/breast/TCGA-BH-A201-01Z-00-DX1.6D6E3224-50A0-45A2-B231-EEF27CA7EFD2.svs": f"{TCGA_REPO_URL}/e70c89a5-1c2f-43f8-b6be-589beea55338",
    "tcga/prostate/TCGA-CH-5753-01A-01-BS1.4311c533-f9c1-4c6f-8b10-922daa3c2e3e.svs": f"{TCGA_REPO_URL}/5a8ce04a-0178-49e2-904c-30e21fb4e41e",
    "tcga/ovarian/TCGA-13-1404-01A-01-TS1.cecf7044-1d29-4d14-b137-821f8d48881e.svs": f"{TCGA_REPO_URL}/e968375e-ef58-4607-b457-e6818b2e8431",
    "9798433/?format=tif": f"{IDR_REPO_URL}/9798433/?format=tif",
    "9798554/?format=tif": f"{IDR_REPO_URL}/9798554/?format=tif",
}

legacy_registry = {
    ("data/" + filename): registry["data/" + filename] for filename in legacy_datasets
}
