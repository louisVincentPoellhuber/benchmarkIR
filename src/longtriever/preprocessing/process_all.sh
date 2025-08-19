# Wikipedia Corpus
echo "Processing Wikipedia corpus."
bash src/longtriever/preprocessing/common/extract_wikipedia.sh

echo "Processing HotPotQA."
python src/longtriever/preprocessing/hotpotqa.py 

echo "Processing NQ."
python src/longtriever/preprocessing/nq.py 

echo "Processing WikIR."
python src/longtriever/preprocessing/wikir.py 

# Articles Corpuses