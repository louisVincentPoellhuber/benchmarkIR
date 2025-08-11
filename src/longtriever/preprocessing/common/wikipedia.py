import wikipediaapi
from common.utils import *
import mwxml
import bz2
import mwparserfromhell
from lxml import etree
import sqlite3
import shutil
BASE_URL = "https://en.wikipedia.org/"

WIKI = wikipediaapi.Wikipedia(USER_AGENT, language='en')  # 'en' for English
    
from line_profiler import profile
import wikiextractor

def download_wikipedia(output_dir, overwrite):
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    zip_out_path = os.path.join(output_dir,"enwiki-latest-pages-articles.xml.bz2")

    if overwrite or not os.path.exists(zip_out_path):
        print("Downloading compressed file.")
        headers = {
            "User-Agent": USER_AGENT
        }
        
        response = requests.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(zip_out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    tqdm_bar.update(len(chunk))
        tqdm_bar.close()


    xml_out_path = os.path.join(output_dir, "enwiki-latest-pages-articles.xml")
    if overwrite or not os.path.exists(xml_out_path):
        print("Extracting compressed file into XML.")
        with bz2.open(zip_out_path, "rb") as source, \
            open(xml_out_path, "wb") as dest:
                shutil.copyfileobj(source, dest)

    return xml_out_path

# def process_pages(title_list, save_path):
    # title_list = set(title_list)
    # nb_docs = len(title_list)
    # nb_found_docs = 0

    # output_path = os.path.join(save_path, "corpus.jsonl")
    # input_path = os.path.join(save_path, "enwiki-latest-pages-articles.xml.bz2")
    # corpus_ids_path = os.path.join(save_path, "corpus_ids.csv")

    # if os.path.exists(corpus_ids_path):
    #     processed_titles = set(pd.read_csv(corpus_ids_path, index_col=0, header=None)[1].to_list())
    #     titles_to_process = processed_titles.symmetric_difference(title_list) 
    # else:
    #     titles_to_process = title_list 
    # redirect_titles = set()

#     print(f"Processing {nb_docs} pages.")
#     for i in range(2):
#         if i==0:
#             titles = titles_to_process
#         elif i==1:
#             titles = redirect_titles
#         with bz2.open(input_path) as dump_file, \
#             open(output_path, "a", encoding="utf-8") as out_file, \
#                 open(corpus_ids_path, "a") as corpus_ids_out:

#             dump = mwxml.Dump.from_file(dump_file)
#             csv_writer = csv.writer(corpus_ids_out)
            
#             with tqdm(total=len(titles)) as pbar:
#                 for page in dump:
#                     title = page.title
#                     if title in titles:
#                         try:
#                             latest_revision = next(page)  # latest only
#                             raw_text = latest_revision.text or ""
#                             wikicode = mwparserfromhell.parse(raw_text)
#                             text = wikicode.strip_code()
#                             if page.redirect is not None:
#                                 redirect_titles.add(page.redirect)
#                                 print(f"Adding '{title}'s' redirect: '{page.redirect}' to list")
#                             else:
#                                 obj = {
#                                     "_id": page.id,
#                                     "title": title,
#                                     "text": text
#                                 }
#                                 out_file.write(json.dumps(obj) + "\n")
#                                 csv_writer.writerow([page.id, title])
#                                 pbar.update(1)
#                                 nb_found_docs += 1
#                         except StopIteration:
#                             continue

#     print(f"Found {nb_found_docs} / {nb_docs} articles.")
    
# def process_pages(title_list, save_path):
#     title_list = set(title_list)
#     nb_docs = len(title_list)
#     nb_found_docs = 0

#     text_buffer = []
#     csv_buffer = []
#     BUFFER_SIZE = 1000  


#     xml_in_path = os.path.join(save_path, "enwiki-latest-pages-articles.xml")
#     corpus_out_path = os.path.join(save_path, "corpus.jsonl")
#     corpus_ids_path = os.path.join(save_path, "corpus_ids.csv")

#     if os.path.exists(corpus_ids_path):
#         processed_titles = set(pd.read_csv(corpus_ids_path, index_col=0, header=None)[1].to_list())
#         titles_to_process = processed_titles.symmetric_difference(title_list) 
#     else:
#         titles_to_process = title_list 
#     redirect_titles = set()    

#     context = etree.iterparse(xml_in_path, events=("start", "end"))

#     # Detect the root tag and extract namespace
#     for event, elem in context:
#         if event == "start":
#             root_tag = elem.tag
#             ns = root_tag.split('}')[0].strip('{')  # Extract namespace URI
#             nsmap = {'ns': ns}
#             break

#     # Reinitialize the context to start parsing properly
#     context = etree.iterparse(xml_in_path, events=("end",), tag=f"{{{ns}}}page")

#     for i in range(2):
#         if i==0:
#             titles = titles_to_process
#         elif i==1:
#             titles = redirect_titles
#         with open(corpus_out_path, "a", encoding="utf-8") as json_out, \
#             open(corpus_ids_path, "a", encoding="utf-8") as csv_out:
#             csv_writer = csv.writer(csv_out)
#             with tqdm(total=len(titles)) as pbar:

#                 for _, elem in context:
#                     title_elem = elem.find("ns:title", namespaces=nsmap)
#                     id_elem = elem.find("ns:id", namespaces=nsmap)  # Page ID (first <id>)
#                     revision_elem = elem.find("ns:revision", namespaces=nsmap)
#                     text_elem = revision_elem.find("ns:text", namespaces=nsmap) if revision_elem is not None else None

#                     if title_elem is not None and text_elem is not None:
#                         title = title_elem.text
                        
#                         if title in titles:
#                             page_id = id_elem.text if id_elem is not None else ""
#                             raw_text = text_elem.text or ""
#                             wikicode = mwparserfromhell.parse(raw_text)
#                             text = wikicode.strip_code()

#                             if text.startswith("#REDIRECT"):
#                                 print(f"Redirect found in title '{title}, text: {text}'")
                            
#                             obj = {
#                                 "_id": page_id,
#                                 "title": title,
#                                 "text": text
#                             }
#                             text_buffer.append(obj)
#                             csv_buffer.append((page_id, title))
#                             titles.remove(title)
#                             pbar.update(1)

#                             if len(text_buffer) >= BUFFER_SIZE:
#                                 print("Saving buffer.")
#                                 json_out.write("\n".join(text_buffer) + "\n")
#                                 text_buffer.clear()

#                                 csv_writer.writerows(csv_buffer)
#                                 csv_buffer.clear()


#                     # Free memory
#                     elem.clear()
#                     while elem.getprevious() is not None:
#                         del elem.getparent()[0]

#     print(f"Found {nb_found_docs} / {nb_docs} articles.")

def create_db(download_dir, db_dir, overwrite):
    db_path = os.path.join(db_dir, "wikipedia.db")

    if overwrite or not os.path.exists(db_path):
        # 1. Create database connection & table
        print("Creating database.")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            text TEXT, 
            url TEXT
        )
        """)
        conn.commit()

        # 2. Walk through every file in every folder
        batch = [] 
        total_processed_pages = 0
        for root, _, files in os.walk(download_dir):
            for filename in files:
                if filename.startswith("wiki_"):
                    filepath = os.path.join(root, filename)

                    # 3. Read file line-by-line (streaming, not loading whole file in memory)
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            article = json.loads(line)

                            batch.append((int(article["id"]), article["title"], article["text"], article["url"]))
                    
                    
                    if len(batch) >= 10000:
                        total_processed_pages+=len(batch)
                        print(f"{total_processed_pages} pages processed.")
                        cur.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", batch)
                        conn.commit()
                        batch.clear()

        total_processed_pages+=len(batch)
        print(f"{total_processed_pages} pages processed.")
        cur.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", batch)
        conn.commit()
        batch.clear()
        
        # 4. Close connection
        conn.close()
    else:
        print("Database already exists. Skipping creation. ")