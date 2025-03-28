echo Syncing...
# rsync -avz --update --progress /data/rech/poellhul/models/new-attention/ $STORAGE_DIR/models/new-attention


dataset=$STORAGE_DIR'/datasets'
batch_size=3
lr=1e-5

# Baseline
exp_name="longformer_test"

echo Training.

model_path=$STORAGE_DIR'/models/new-attention/dpr/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

model_config='{"q_model": "allenai/longformer-base-4096",
        "doc_model": "allenai/longformer-base-4096",
        "shared_encoder": false,
        "normalize": false, 
        "attn_implementation": "eager", 
        "query_prompt": "",
        "passage_prompt": "", 
        "sep": " [SEP] "
        }'


config='{"settings": {
        "task": "nq",
        "exp_name": "'$exp_name'",
        "save_path": "'$model_path'",
        "logging": true,
        "accelerate": true,
        "eval_hf_model": false,
        "negatives": false,
        "epochs": 5,
        "batch_size": '$batch_size',  
        "lr": '$lr'
        },
        "config":'$model_config'}'

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 
# accelerate launch src/retrieval/train_dpr.py --config_dict "$config"
# python src/retrieval/train_dpr.py --config_dict "$config"

python src/retrieval/evaluate_dpr.py  --config_dict "$config"