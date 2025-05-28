export WANDB_DISABLED=true
export COMET_LOG_ASSETS=true
export COMET_API_KEY=TzEzoqltg1eu3XaFzpKHYuQaD
python -m torch.distributed.launch src/longtriever/run.py \
        --corpus_file /Tmp/lvpoellhuber/datasets/msmarco-doc/corpus.jsonl \
        --qrels_file /Tmp/lvpoellhuber/datasets/msmarco-doc/qrels/train.tsv \
        --query_file /Tmp/lvpoellhuber/datasets/msmarco-doc/queries.jsonl \
        --max_query_length 512 \
        --max_corpus_length 512 \
        --max_corpus_sent_num 8 \
        --model_type longtriever \
        --model_name_or_path /Tmp/lvpoellhuber/models/longtriever/pretrained/bert-base-uncased \
        --output_dir ./output_longtriever_finetune_long2 \
        --do_train True \
        --num_train_epochs 1 \
        --save_strategy epoch \
        --per_device_train_batch_size 5 \
        --dataloader_drop_last True \
        --fp16 False \
        --learning_rate 1e-4 \
        --overwrite_output_dir False \
        --dataloader_num_workers 12 \
        --disable_tqdm False \
        --report_to comet_ml \
        --logging_steps 100 