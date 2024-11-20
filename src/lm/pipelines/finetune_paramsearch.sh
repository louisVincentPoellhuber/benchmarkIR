################################### RoBERTa 0 ###################################
model_config='{"max_position_embeddings": 512,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "eager",
        "num_labels": 2}'
# batch_size = 44
# lr = 1e-5

# It's already been done! 
# python src/lm/metrics.py --path "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/roberta_0"


################################### RoBERTa 1 ###################################

batch_size=44
lr=1e-4

exp_name="roberta_1"

echo Training Roberta 1.

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'



config='{"settings": {
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
            "lr": '$lr'},
        "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Roberta 1.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path

################################### Adaptive 0 ###################################
model_config='{
        "max_position_embeddings": 512,
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
lr=1e-5

exp_name="adaptive_0"

echo Training Adaptive 0.

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

config='{"settings": {
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
            "lr": '$lr'},
        "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Adaptive 0.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path


################################### Adaptive 1 ###################################
batch_size=44
lr=1e-4

exp_name="adaptive_1"

echo Training Adaptive 1.

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

config='{"settings": {
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
            "lr": '$lr'},
        "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Adaptive 1.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path


################################### Sparse 0 ###################################

model_config='{
        "vocab_size": 32,
        "max_position_embeddings": 512,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'

    
batch_size=44
lr=1e-5

exp_name="sparse_0"

echo Training Sparse 0.

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

config='{"settings": {
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
            "lr": '$lr'},
        "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Sparse 0.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path


################################### Sparse 1 ###################################
# batch_size = 44
# lr = 1e-4

batch_size=44
lr=1e-4

exp_name="sparse_1"

echo Training Sparse 1.

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

config='{"settings": {
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
            "lr": '$lr'},
        "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Sparse 1.


python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path

