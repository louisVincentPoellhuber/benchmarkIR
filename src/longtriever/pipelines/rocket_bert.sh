task="msmarco-passage"

psg_exp_name="rocket_bert_psg"
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
        --model_type bert \
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
        --logging_steps 100 

        
task="msmarco-doc"

exp_name="rocket_bert_doc"
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
        --model_type bert\
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
        --logging_steps 100 

echo Evaluating.
python src/longtriever/evaluate_longtriever.py \
        --task $task \
        --corpus_file $STORAGE_DIR/datasets/$task/corpus.jsonl \
        --qrels_file $STORAGE_DIR/datasets/$task/qrels/test.tsv \
        --query_file $STORAGE_DIR/datasets/$task/queries.jsonl \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --model_type bert\
        --model_name_or_path $model_path \
        --output_dir $model_path \
        --do_train False \
        --num_train_epochs 3 \
        --save_strategy epoch \
        --per_device_train_batch_size 10 \
        --per_device_eval_batch_size 10 \
        --dataloader_drop_last True \
        --fp16 True \
        --learning_rate 1e-4 \
        --overwrite_output_dir True \
        --dataloader_num_workers 12 \
        --disable_tqdm False \
        --report_to comet_ml \
        --run_name $exp_name\
        --logging_steps 100
