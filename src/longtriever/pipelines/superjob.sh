bash src/retrieval/pipelines/preprocess_msmarco.sh 

bash src/longtriever/pipelines/rocket_og_hierarchical.sh
bash src/longtriever/pipelines/rocket_og_hierarchical.sh
bash src/longtriever/pipelines/rocket_long_msmarco.sh # BERT gave zeroes
bash src/longtriever/pipelines/reverse_ablation.sh 
bash src/longtriever/pipelines/lt_block_size.sh 
bash src/longtriever/pipelines/og_longtriever_epochs.sh # Re-test more directly
bash src/longtriever/pipelines/negative_lt.sh 

bash src/longtriever/pipelines/superjob+.sh # This script is to add any tests after this script is launched. 