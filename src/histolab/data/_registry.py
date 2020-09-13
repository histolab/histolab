# flake8: noqa

# in legacy datasets we need to put our sample data within the data dir
legacy_datasets = ["cmu_small_region.svs"]

# Registry of datafiles that can be downloaded along with their SHA256 hashes
# To generate the SHA256 hash, use the command
# openssl sha256 filename
registry = {
    "data/cmu_small_region.svs": "ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7",
    "aperio/JP2K-33003-1.svs": "6205ccf75a8fa6c32df7c5c04b7377398971a490fb6b320d50d91f7ba6a0e6fd",
    "aperio/JP2K-33003-2.svs": "1a13cef86b55b51127cebd94a1f6069f7de494c98e3e708640d1ce7181d9e3fd",
    "tcga/breast/9c960533-2e58-4e54-97b2-8454dfb4b8c8": "03f542afa2d70224d594b2cca33b99977a5c0e41b1a8d03471ab3cf62ea3c4b3",
    "tcga/breast/da36d3aa-9b19-492a-af4f-cc028a926d96": "2172cca68a8b7722d281174a74c4f112d0f52fc71710d7d605f401731c783fc9",
    "tcga/breast/f8b4cee6-9149-45b4-ae53-82b0547e1e34": "55c694262c4d44b342e08eb3ef2082eeb9e9deeb3cb445e4776419bb9fa7dc21",
    "tcga/breast/31e248bf-ee24-4d18-bccb-47046fccb461": "95163831d9076bb5e5b21790933dee9535a3607ba35bd6ae425374a45ecb1ba6",
    "tcga/prostate/6b725022-f1d5-4672-8c6c-de8140345210": "305c80e28227b25fdd0cc24726da4cf038380b4326e25c6518ffe23051a25ac0",
    "tcga/ovarian/b777ec99-2811-4aa4-9568-13f68e380c86": "f8e5059a0c9f8c026cfb2613cddef6562f8cdbd5954580282e2afa41d2f86a8c",
}

APERIO_REPO_URL = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio"
TCGA_REPO_URL = "https://api.gdc.cancer.gov/data"

registry_urls = {
    "aperio/JP2K-33003-1.svs": f"{APERIO_REPO_URL}/JP2K-33003-1.svs",
    "aperio/JP2K-33003-2.svs": f"{APERIO_REPO_URL}/JP2K-33003-2.svs",
    "tcga/breast/9c960533-2e58-4e54-97b2-8454dfb4b8c8": f"{TCGA_REPO_URL}/9c960533-2e58-4e54-97b2-8454dfb4b8c8",
    "tcga/breast/da36d3aa-9b19-492a-af4f-cc028a926d96": f"{TCGA_REPO_URL}/da36d3aa-9b19-492a-af4f-cc028a926d96",
    "tcga/breast/f8b4cee6-9149-45b4-ae53-82b0547e1e34": f"{TCGA_REPO_URL}/f8b4cee6-9149-45b4-ae53-82b0547e1e34",
    "tcga/breast/31e248bf-ee24-4d18-bccb-47046fccb461": f"{TCGA_REPO_URL}/31e248bf-ee24-4d18-bccb-47046fccb461",
    "tcga/prostate/6b725022-f1d5-4672-8c6c-de8140345210": f"{TCGA_REPO_URL}/6b725022-f1d5-4672-8c6c-de8140345210",
    "tcga/ovarian/b777ec99-2811-4aa4-9568-13f68e380c86": f"{TCGA_REPO_URL}/b777ec99-2811-4aa4-9568-13f68e380c86",
}

legacy_registry = {
    ("data/" + filename): registry["data/" + filename] for filename in legacy_datasets
}
