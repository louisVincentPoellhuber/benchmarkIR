import os
import sys
from beir import util


STORAGE_DIR = os.getenv("STORAGE_DIR")

# First do Test, in case it hasn't been done yet
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
data_path = util.download_and_unzip(url, STORAGE_DIR+"/datasets/nq")
