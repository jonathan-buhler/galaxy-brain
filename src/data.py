from config.keys import username, key

import os
import shutil
from zipfile import ZipFile


os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key

data_dir = "./src/data"
downloads_dir = "downloads"
download_file = "galaxy-zoo-the-galaxy-challenge.zip"
images_file = "images_training_rev1.zip"


def fetch_data():
    # Import is here because environment variables need to be set before importing
    from kaggle.api import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # Downloads zipfile
    print("Downloading dataset...")
    api.competition_download_files(
        "galaxy-zoo-the-galaxy-challenge", path=f"{data_dir}/{downloads_dir}"
    )

    # Unzips zipfile into new folder
    print("Unzipping files...")
    with ZipFile(f"{data_dir}/{downloads_dir}/{download_file}", "r") as download:
        download.extractall(f"{data_dir}/{downloads_dir}")

    # Unzips image zipfile
    with ZipFile(f"{data_dir}/{downloads_dir}/{images_file}", "r") as data:
        data.extractall(f"{data_dir}")

    # Renames images directory
    print("Cleaning up...")
    os.rename(f"{data_dir}/images_training_rev1", f"{data_dir}/images")

    # Deletes other downloaded files
    shutil.rmtree(f"{data_dir}/{downloads_dir}")
