import requests
from tqdm import tqdm
import os

def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with (
        open(save_path, "wb") as fd,
        tqdm(
            desc=save_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar,
    ):
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)


files = ["msmarco-docs.tsv.gz", "msmarco-docs-lookup.tsv.gz", "msmarco-doctrain-queries.tsv.gz", "msmarco-doctrain-top100.gz", "msmarco-doctrain-qrels.tsv.gz", "msmarco-doctriples.py", "msmarco-docdev-queries.tsv.gz", "msmarco-docdev-top100.gz", "msmarco-docdev-qrels.tsv.gz", "docleaderboard-queries.tsv.gz", "docleaderboard-top100.gz"]
url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
save_dir = "/Tmp/lvpoellhuber/datasets/msmarco-doc"

for download_file in files:
    file_url = url + download_file
    save_path = os.path.join(save_dir, download_file)
    
    if os.path.exists(save_path):
        print("Download already exists. ")
    else:
        download_url(url = file_url, save_path = save_path)