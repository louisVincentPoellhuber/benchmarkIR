
bash src/longtriever/pipelines/rocket_og_hierarchical.sh # Still need to fix inter-block problems 
bash src/longtriever/pipelines/speed_and_init_test.sh
bash src/longtriever/pipelines/linear_scheduler.sh 
bash src/longtriever/pipelines/reverse_ablation.sh # Still need to fix inter-block problems 
bash src/longtriever/pipelines/rocket_long_msmarco.sh # BERT gave zeroes
bash src/longtriever/pipelines/lt_block_size.sh 
bash src/longtriever/pipelines/rocket_bert.sh # Crashed
bash src/longtriever/pipelines/og_longtriever_epochs.sh # Re-test more directly

bash src/longtriever/pipelines/superjob+.sh # This script is to add any tests after this script is launched. 