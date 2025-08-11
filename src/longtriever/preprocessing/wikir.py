import os
import requests
from tqdm import tqdm

LOCAL_PATH = "/Tmp/lvpoellhuber/datasets/wikir/"
os.makedirs(LOCAL_PATH, exist_ok=True)

file_url = "https://zenodo.org/records/3707238/files/enwikIRS.zip?download=1"

local_path = os.path.join(LOCAL_PATH, "DORIS-enwikIRS.zip")

with requests.get(file_url, stream=True) as r:
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=8192)):
            f.write(chunk) 