echo Syncing...
# rsync -avz --update --progress /data/rech/poellhul/models/new-attention/ $STORAGE_DIR/models/new-attention

dataset=$STORAGE_DIR'/datasets'
batch_size=3
lr=1e-4
exp_name="longtriever_eval_short"
export EXP_NAME=$exp_name

model_path=$STORAGE_DIR'/models/longtriever/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi
dataset=$STORAGE_DIR'/datasets'

model_config='{"q_model": "STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased",
        "doc_model": "STORAGE_DIR/models/longtriever/pretrained/bert-base-uncased",
        "shared_encoder": false,
        "normalize": false, 
        "attn_implementation": "eager", 
        "query_prompt": "",
        "passage_prompt": "", 
        "sep": " [SEP] ", 
        "max_block_length": 512, 
        "max_num_blocks": 8
        }'

config='{"settings": {
        "task": "msmarco-doc-short",
        "exp_name": "'$exp_name'",
        "save_path": "'$model_path'",
        "logging": true,
        "accelerate": true,
        "eval_hf_model": false,
        "negatives": false,
        "epochs": 1,
        "batch_size": '$batch_size',  
        "lr": '$lr'
        },
        "config":'$model_config'}'

        
echo Evaluating. 
python src/retrieval/evaluate_longtriever.py  --config_dict "$config" 