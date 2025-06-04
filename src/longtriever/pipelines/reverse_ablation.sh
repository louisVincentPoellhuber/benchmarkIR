task="msmarco-doc"

exp_name="revabl_no_interblock"
model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi
train_batch_size=10

export EXP_NAME=$exp_name

echo Training on documents.
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 1 \
#         --model_type longtriever \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 1 \
#         --save_strategy epoch \
#         --per_device_train_batch_size $train_batch_size \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --logging_steps 100 \
#         --run_name $exp_name\
#         --ablation_config '{"inter_block_encoder": false, "doc_token": false}'

# echo Evaluating.
# rm -f $model_path'/mprofile.dat'
# mprof run --output $model_path'/mprofile.dat' src/longtriever/evaluate_longtriever.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 1 \
#         --model_type longtriever\
#         --model_name_or_path $model_path \
#         --output_dir $model_path \
#         --do_train False \
#         --num_train_epochs 1 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --per_device_eval_batch_size 32 \
#         --dataloader_drop_last True \
#         --fp16 True \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder": false, "doc_token": false}'
# mprof plot --output $model_path'/memory.png'

exp_name="revabl_doctoken"
model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

export EXP_NAME=$exp_name

echo Training on documents.
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 1 \
#         --model_type longtriever \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 1 \
#         --save_strategy epoch \
#         --per_device_train_batch_size $train_batch_size \
#         --dataloader_drop_last True \
#         --fp16 True \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --logging_steps 100 \
#         --run_name $exp_name\
#         --ablation_config '{"inter_block_encoder": false, "doc_token": true}'

echo Evaluating.
rm -f $model_path'/mprofile.dat'

# mprof run --output $model_path'/mprofile.dat' src/longtriever/evaluate_longtriever.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 1 \
#         --model_type longtriever\
#         --model_name_or_path $model_path \
#         --output_dir $model_path \
#         --do_train False \
#         --num_train_epochs 1 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --per_device_eval_batch_size 32 \
#         --dataloader_drop_last True \
#         --fp16 True \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder": false, "doc_token": true}'
# mprof plot --output $model_path'/memory.png'


exp_name="revabl_short_interblock"
model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

export EXP_NAME=$exp_name

# echo Training on documents.
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 1 \
#         --model_type longtriever \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 1 \
#         --save_strategy epoch \
#         --per_device_train_batch_size $train_batch_size \
#         --dataloader_drop_last True \
#         --fp16 True \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --logging_steps 100 \
#         --run_name $exp_name\
#         --ablation_config '{"inter_block_encoder": true, "doc_token": true}'

echo Evaluating.
python src/longtriever/evaluate_longtriever.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
        --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 1 \
        --model_type longtriever\
        --model_name_or_path $model_path \
        --output_dir $model_path \
        --do_train False \
        --num_train_epochs 1 \
        --save_strategy epoch \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 32 \
        --dataloader_drop_last True \
        --fp16 True \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 12 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100 \
        --ablation_config '{"inter_block_encoder": true, "doc_token": true}'



exp_name="revabl_regular"
model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

export EXP_NAME=$exp_name

echo Training on documents.
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type longtriever \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 1 \
#         --save_strategy epoch \
#         --per_device_train_batch_size $train_batch_size \
#         --dataloader_drop_last True \
#         --fp16 True \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --logging_steps 100 \
#         --run_name $exp_name\
#         --ablation_config '{"inter_block_encoder": true, "doc_token": true}'

echo Evaluating.
python src/longtriever/evaluate_longtriever.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
        --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type longtriever\
        --model_name_or_path $model_path \
        --output_dir $model_path \
        --do_train False \
        --num_train_epochs 1 \
        --save_strategy epoch \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 32 \
        --dataloader_drop_last True \
        --fp16 True \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 12 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100 \
        --ablation_config '{"inter_block_encoder": true, "doc_token": true}'
