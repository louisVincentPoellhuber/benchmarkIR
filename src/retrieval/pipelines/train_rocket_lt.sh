echo Syncing...

dataset=$STORAGE_DIR'/datasets'
lr=1e-4
dataset=$STORAGE_DIR'/datasets'


exp_name="rocket_lt-passage"
batch_size=44

model_path=$STORAGE_DIR'/models/longtriever/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

echo Preprocessing data. 
python src/retrieval/preprocessing/preprocess_msmarco-passage.py 

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
        "task": "msmarco-passage",
        "exp_name": "'$exp_name'",
        "save_path": "'$model_path'",
        "logging": true,
        "accelerate": true,
        "eval_hf_model": false,
        "negatives": false,
        "epochs": 3,
        "batch_size": '$batch_size',  
        "lr": '$lr'
        },
        "config":'$model_config'}'

        
echo Initializing model.
python src/retrieval/init_longtriever.py --config_dict "$config" --base_model "google-bert/bert-base-uncased"

echo Training.
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
NCCL_DEBUG=WARN TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch src/retrieval/train_longtriever.py --config_dict "$config"


exp_name="rocket_lt-doc"
batch_size=3
lr=3e-5

model_path=$STORAGE_DIR'/models/longtriever/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

echo Preprocessing data. 
python src/retrieval/preprocessing/preprocess_msmarco-passage.py 
model_config='{"q_model": "STORAGE_DIR/models/longtriever/rocket_lt-passage/q_model",
        "doc_model": "STORAGE_DIR/models/longtriever/rocket_lt-passage/doc_model",
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
        "task": "msmarco-doc",
        "exp_name": "'$exp_name'",
        "save_path": "'$model_path'",
        "logging": true,
        "accelerate": true,
        "eval_hf_model": false,
        "negatives": false,
        "epochs": 3,
        "batch_size": '$batch_size',  
        "lr": '$lr'
        },
        "config":'$model_config'}'

        
echo Initializing model.
python src/retrieval/init_longtriever.py --config_dict "$config" --base_model "google-bert/bert-base-uncased"

echo Training.
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
NCCL_DEBUG=WARN TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch src/retrieval/train_longtriever.py --config_dict "$config"

echo Evaluating. 
rm -f $model_path'/mprofile.dat'
mprof run --output $model_path'/mprofile.dat' src/retrieval/evaluate_longtriever.py  --config_dict "$config" 
mprof plot --output $model_path'/memory.png' $model_path'/mprofile.dat'

rsync -avz --update --progress $model_path /data/rech/poellhul/models/longtriever/
