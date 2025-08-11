# bash src/retrieval/pipelines/preprocess_msmarco.sh 




task="msmarco-passage"

psg_exp_name="hier_ablation_passage-analyzed"
psg_model_path=$STORAGE_DIR'/models/longtriever_og/'$psg_exp_name
echo $psg_model_path
if [[ ! -d $psg_model_path ]]; then
  mkdir -p $psg_model_path
fi
export EXP_NAME=$psg_exp_name

echo Training on passages. 
torchrun --nproc_per_node=4 src/longtriever/run.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
        --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type hierarchical \
        --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
        --output_dir $psg_model_path \
        --do_train True \
        --num_train_epochs 3 \
        --save_strategy epoch \
        --per_device_train_batch_size 96 \
        --dataloader_drop_last True \
        --fp16 False \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 12 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $psg_exp_name\
        --logging_steps 100 \
        --ablation_config '{"inter_block_encoder":true, "doc_token":false, "segments": true, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "first"}'

task="msmarco-doc"

exp_name="hier_ablation_doc-analyzed"
model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi
export EXP_NAME=$exp_name

echo Training on documents.
torchrun --nproc_per_node=4 src/longtriever/run.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
        --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type hierarchical \
        --model_name_or_path $psg_model_path \
        --output_dir $model_path \
        --do_train True \
        --num_train_epochs 3 \
        --save_strategy epoch \
        --per_device_train_batch_size 10 \
        --dataloader_drop_last True \
        --fp16 False \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 12 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100 \
        --ablation_config '{"inter_block_encoder":true, "doc_token":false, "segments": true, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "first"}'


echo Evaluating.

python src/longtriever/evaluate_longtriever.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
        --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type hierarchical\
        --model_name_or_path $model_path \
        --output_dir $model_path \
        --do_train False \
        --num_train_epochs 3 \
        --save_strategy epoch \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 32 \
        --dataloader_drop_last True \
        --fp16 False \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 12 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100 \
        --ablation_config '{"inter_block_encoder":true, "doc_token":false, "segments": true, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "first"}'





# echo Init LT.
# python src/longtriever/init_longtriever.py \
#         --model_type longtriever\
#         --overwrite_output_dir True \
#         --base_model google-bert/bert-base-uncased \
#         --base_model_postfix True

# task="msmarco-passage"

# psg_exp_name="hier_ablation_passage-sep"
# psg_model_path=$STORAGE_DIR'/models/longtriever_og/'$psg_exp_name
# echo $psg_model_path
# if [[ ! -d $psg_model_path ]]; then
#   mkdir -p $psg_model_path
# fi
# export EXP_NAME=$psg_exp_name

# echo Training on passages. 
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $psg_model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 96 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $psg_exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": true, "text_separator": true, "end_separator": true, "cls_position": "first"}'

# task="msmarco-doc"

# exp_name="hier_ablation_doc-sep"
# model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
# echo $model_path
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi
# export EXP_NAME=$exp_name

# echo Training on documents.
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $psg_model_path \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": true, "text_separator": true, "end_separator": true, "cls_position": "first"}'


# echo Evaluating.

# python src/longtriever/evaluate_longtriever.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical\
#         --model_name_or_path $model_path \
#         --output_dir $model_path \
#         --do_train False \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --per_device_eval_batch_size 32 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": true, "text_separator": true, "end_separator": true, "cls_position": "first"}'



# task="msmarco-passage"

# psg_exp_name="hier_ablation_passage-seg"
# psg_model_path=$STORAGE_DIR'/models/longtriever_og/'$psg_exp_name
# echo $psg_model_path
# if [[ ! -d $psg_model_path ]]; then
#   mkdir -p $psg_model_path
# fi
# export EXP_NAME=$psg_exp_name

# echo Training on passages. 
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $psg_model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 96 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $psg_exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": true, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "first"}'

# task="msmarco-doc"

# exp_name="hier_ablation_doc-seg"
# model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
# echo $model_path
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi
# export EXP_NAME=$exp_name

# echo Training on documents.
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $psg_model_path \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": true, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "first"}'


# echo Evaluating.

# python src/longtriever/evaluate_longtriever.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical\
#         --model_name_or_path $model_path \
#         --output_dir $model_path \
#         --do_train False \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --per_device_eval_batch_size 32 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": true, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "first"}'

# task="msmarco-passage"



# psg_exp_name="hier_ablation_passage-relative_cls"
# psg_model_path=$STORAGE_DIR'/models/longtriever_og/'$psg_exp_name
# echo $psg_model_path
# if [[ ! -d $psg_model_path ]]; then
#   mkdir -p $psg_model_path
# fi
# export EXP_NAME=$psg_exp_name

# echo Training on passages. 
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $psg_model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 96 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $psg_exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "relative"}'

# task="msmarco-doc"

# exp_name="hier_ablation_doc-relative_cls"
# model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
# echo $model_path
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi
# export EXP_NAME=$exp_name

# echo Training on documents.
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $psg_model_path \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "relative"}'


# echo Evaluating.

# python src/longtriever/evaluate_longtriever.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical\
#         --model_name_or_path $model_path \
#         --output_dir $model_path \
#         --do_train False \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --per_device_eval_batch_size 32 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "relative"}'




# task="msmarco-passage"

# psg_exp_name="hier_ablation_passage-original"
# psg_model_path=$STORAGE_DIR'/models/longtriever_og/'$psg_exp_name
# echo $psg_model_path
# if [[ ! -d $psg_model_path ]]; then
#   mkdir -p $psg_model_path
# fi
# export EXP_NAME=$psg_exp_name

# echo Training on passages. 
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $psg_model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 96 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $psg_exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "relative"}'

# task="msmarco-doc"

# exp_name="hier_ablation_doc-original"
# model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
# echo $model_path
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi
# export EXP_NAME=$exp_name

# echo Training on documents.
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $psg_model_path \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "relative"}'


# echo Evaluating.

# python src/longtriever/evaluate_longtriever.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical\
#         --model_name_or_path $model_path \
#         --output_dir $model_path \
#         --do_train False \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --per_device_eval_batch_size 32 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "relative"}'


# task="msmarco-passage"

# psg_exp_name="hier_ablation_passage-start_sep"
# psg_model_path=$STORAGE_DIR'/models/longtriever_og/'$psg_exp_name
# echo $psg_model_path
# if [[ ! -d $psg_model_path ]]; then
#   mkdir -p $psg_model_path
# fi
# export EXP_NAME=$psg_exp_name

# echo Training on passages. 
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $psg_model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 96 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $psg_exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": true, "text_separator": true, "end_separator": false, "cls_position": "relative"}'

task="msmarco-doc"

exp_name="hier_ablation_doc-start_sep"
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
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $psg_model_path \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": true, "text_separator": true, "end_separator": false, "cls_position": "relative"}'


echo Evaluating.

python src/longtriever/evaluate_longtriever.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
        --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type hierarchical\
        --model_name_or_path $model_path \
        --output_dir $model_path \
        --do_train False \
        --num_train_epochs 3 \
        --save_strategy epoch \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 32 \
        --dataloader_drop_last True \
        --fp16 False \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 12 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100 \
        --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "relative"}'



# task="msmarco-passage"

# psg_exp_name="hier_ablation_passage-end_sep"
# psg_model_path=$STORAGE_DIR'/models/longtriever_og/'$psg_exp_name
# echo $psg_model_path
# if [[ ! -d $psg_model_path ]]; then
#   mkdir -p $psg_model_path
# fi
# export EXP_NAME=$psg_exp_name

# echo Training on passages. 
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $psg_model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 96 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $psg_exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": true, "cls_position": "relative"}'

task="msmarco-doc"

exp_name="hier_ablation_doc-end_sep"
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
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $psg_model_path \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": true, "cls_position": "relative"}'


echo Evaluating.

python src/longtriever/evaluate_longtriever.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
        --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type hierarchical\
        --model_name_or_path $model_path \
        --output_dir $model_path \
        --do_train False \
        --num_train_epochs 3 \
        --save_strategy epoch \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 32 \
        --dataloader_drop_last True \
        --fp16 False \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 12 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100 \
        --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": true, "end_separator": false, "cls_position": "relative"}'


# task="msmarco-passage"

# psg_exp_name="hier_ablation_passage-start_end_sep"
# psg_model_path=$STORAGE_DIR'/models/longtriever_og/'$psg_exp_name
# echo $psg_model_path
# if [[ ! -d $psg_model_path ]]; then
#   mkdir -p $psg_model_path
# fi
# export EXP_NAME=$psg_exp_name

# echo Training on passages. 
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased \
#         --output_dir $psg_model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 96 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $psg_exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": true, "text_separator": false, "end_separator": true, "cls_position": "relative"}'

# task="msmarco-doc"

# exp_name="hier_ablation_doc-start_end_sep"
# model_path=$STORAGE_DIR'/models/longtriever_og/'$exp_name
# echo $model_path
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi
# export EXP_NAME=$exp_name

# echo Training on documents.
# torchrun --nproc_per_node=4 src/longtriever/run.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/train.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical \
#         --model_name_or_path $psg_model_path \
#         --output_dir $model_path \
#         --do_train True \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": true, "text_separator": false, "end_separator": true, "cls_position": "relative"}'


# echo Evaluating.

# python src/longtriever/evaluate_longtriever.py \
#         --task $task \
#         --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
#         --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
#         --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
#         --max_query_length 512 \
#         --max_corpus_length 512 \
#         --max_corpus_sent_num 8 \
#         --model_type hierarchical\
#         --model_name_or_path $model_path \
#         --output_dir $model_path \
#         --do_train False \
#         --num_train_epochs 3 \
#         --save_strategy epoch \
#         --per_device_train_batch_size 10 \
#         --per_device_eval_batch_size 32 \
#         --dataloader_drop_last True \
#         --fp16 False \
#         --learning_rate 1e-4 \
#         --overwrite_output_dir True \
#         --dataloader_num_workers 12 \
#         --disable_tqdm False \
#         --report_to comet_ml \
#         --run_name $exp_name\
#         --logging_steps 100 \
#         --ablation_config '{"inter_block_encoder":true, "doc_token":true, "segments": false, "start_separator": false, "text_separator": false, "end_separator": false, "cls_position": "relative"}'
