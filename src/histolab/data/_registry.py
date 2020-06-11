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
}

aperio_repo_url = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio"
tcga_repo_url = "https://api.gdc.cancer.gov/data"

registry_urls = {
    "aperio/JP2K-33003-1.svs": f"{aperio_repo_url}/JP2K-33003-1.svs",
    "aperio/JP2K-33003-2.svs": f"{aperio_repo_url}/JP2K-33003-2.svs",
    "tcga/breast/9c960533-2e58-4e54-97b2-8454dfb4b8c8": f"{tcga_repo_url}/9c960533-2e58-4e54-97b2-8454dfb4b8c8",
}

legacy_registry = {
    ("data/" + filename): registry["data/" + filename] for filename in legacy_datasets
}
