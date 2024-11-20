
################################### Adaptive ###################################
# Same for all adpative models, for now
model_config='{
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "adaptive",
        "num_labels":4,
        "inner_hidden_size": 1024,
        "dropout": 0,
        "attn_span": 1024,
        "adapt_span_enabled": true,
        "adapt_span_loss": 2e-06,
        "adapt_span_ramp": 32,
        "adapt_span_init": 0,
        "adapt_span_cache": true
    }'
batch_size=44
lr=1e-4
dataset=$STORAGE_DIR'/datasets'


echo Training Adaptive LR 5e-4.
adaptive_lr=5e-4 # only variable that changes for this particular tuning

exp_name="adaptive_lr_5e4"


model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi



config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": false,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "adaptive_lr":'$adaptive_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"


echo Evaluating Adaptive 5e-4.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path


echo Training Adaptive LR 1e-3.
adaptive_lr=1e-3

exp_name="adaptive_lr_1e3"

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": false,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "adaptive_lr":'$adaptive_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Adaptive LR 1e-3.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path


echo Training Adaptive LR 1e-2.

adaptive_lr=1e-2
exp_name="adaptive_lr_1e2"

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": false,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "adaptive_lr":'$adaptive_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Adaptive 1e-2.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path



################################### Sparse 1 ###################################
# batch_size = 44
# lr = 1e-4
# those two stay the same as adaptive

model_config='{
        "vocab_size": 32,
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'


echo Training Sparse LR 1.

alpha_lr=1
exp_name="sparse_lr_1"

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": false,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Sparse LR 1.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path


echo Training Sparse LR 2.

alpha_lr=2
exp_name="sparse_lr_2"

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": false,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Sparse LR 2.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path



echo Training Sparse LR 5.

alpha_lr=5
exp_name="sparse_lr_5"

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": false,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Sparse LR 5.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path



echo Training Sparse LR 15.

alpha_lr=15
exp_name="sparse_lr_15"

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": false,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Sparse LR 15.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path
