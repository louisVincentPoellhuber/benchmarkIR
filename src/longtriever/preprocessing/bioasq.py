import json
import os
import re
import requests
import wget
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import dotenv
dotenv.load_dotenv()

LOCAL_PATH = "/Tmp/lvpoellhuber/datasets/bioasq/"
os.makedirs(LOCAL_PATH, exist_ok=True)

pubmed_directories = [
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/",
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/xml/"
]


for directory in pubmed_directories:
    response = requests.get(directory)
    soup = BeautifulSoup(response.text, "html.parser")

    # Step 2: Extract all .tar.gz links
    links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.tar.gz')]

    # Step 3: Download each file
    for link in tqdm(links, desc="Downloading files"):
        file_url = urljoin(directory, link)
        local_path = os.path.join(LOCAL_PATH, link)

        if os.path.exists(local_path):
            print(f"Already downloaded: {link}")
            continue

        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)