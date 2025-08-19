from common.utils import * 

def download_s2orc(download_dir, overwrite=False):
    if len(os.listdir(download_dir)) ==0 or overwrite:
        API_KEY = os.getenv("S2_API_KEY")
        if not API_KEY:
            raise ValueError("Please set the S2_API_KEY environment variable with your Semantic Scholar API key.")

        # get latest release's ID
        response = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
        RELEASE_ID = response["release_id"]
        print(f"Latest release ID: {RELEASE_ID}")

        # get the download links for the s2orc dataset; needs to pass API key through `x-api-key` header
        # download via wget. this can take a while...
        response = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/s2orc/", headers={"x-api-key": API_KEY}).json()
        for url in tqdm(response["files"]):
            match = re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url)
            assert match.group(1) == RELEASE_ID
            SHARD_ID = match.group(2)
            wget.download(url, out=os.path.join(download_dir, f"{SHARD_ID}.gz"))
        print("Downloaded all shards.")
    else:
        log_message("S2ORC shards already exist. Skipping download.", print_message=True)

def create_db(download_dir, db_dir, overwrite):
    db_path = os.path.join(db_dir, "corpus.db")
    shards_path = os.listdir(download_dir)

    if overwrite or not os.path.exists(db_path):
        # 1. Create database connection & table
        print("Creating database.")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT  PRIMARY KEY,
            corpusid INTEGER,
            text TEXT
        )
        """)
        conn.commit()

        # 2. Walk through every file in every folder
        batch = [] 
        total_processed_pages = 0
        for shard in tqdm(shards_path):
            shard = os.path.join('/Tmp/lvpoellhuber/datasets/vault/corpus/s2orc/downloads', shard)
            with gzip.open(shard, 'rt', encoding='utf-8') as f:
                for line in tqdm(f):
                    line = line.strip()
                    article = json.loads(line)

                    ssid = article["content"]["source"]["pdfsha"]
                    text = article["content"]["text"]
                    corpusid = article["corpusid"]
                    
                    batch.append((ssid, text, corpusid))
        
        total_processed_pages+=len(batch)
        cur.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", batch)
        conn.commit()
        batch.clear()

        # 4. Close connection
        conn.close()
    else:
        print("Database already exists. Skipping creation. ")