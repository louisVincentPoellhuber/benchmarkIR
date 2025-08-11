
curl -L --progress-bar -C - https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -o /Tmp/lvpoellhuber/datasets/belt/corpus/wikipedia/downloads

python wikiextractor/WikiExtractor.py --json --processes 16 --output /Tmp/lvpoellhuber/datasets/belt/corpus/wikipedia/downloads /Tmp/lvpoellhuber/datasets/belt/corpus/wikipedia/downloads/enwiki-latest-pages-articles.xml.bz2