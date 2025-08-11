import os
import requests
from tqdm import tqdm

LOCAL_PATH = "/Tmp/lvpoellhuber/datasets/cord19/"
os.makedirs(LOCAL_PATH, exist_ok=True)

# There are more URLs
file_url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-04-10.tar.gz"

local_path = os.path.join(LOCAL_PATH, "cord-19_2020-04-10.tar.gz")

with requests.get(file_url, stream=True) as r:
    r.raise_for_status()
    with open(local_path, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=8192)):
            f.write(chunk) 