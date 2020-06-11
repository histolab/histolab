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
}

aperio_repo_url = "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/"

registry_urls = {
    "aperio/JP2K-33003-1.svs": f"{aperio_repo_url}JP2K-33003-1.svs",
    "aperio/JP2K-33003-2.svs": f"{aperio_repo_url}JP2K-33003-2.svs",
}

legacy_registry = {
    ("data/" + filename): registry["data/" + filename] for filename in legacy_datasets
}
