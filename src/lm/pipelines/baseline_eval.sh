echo $STORAGE_DIR

################################### RoBERTa Baseline ###################################
batch_size=44
lr=1e-4

exp_name="roberta_baseline"

echo Evaluating Roberta Baseline.

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

model_config='{"max_position_embeddings": 512,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "eager",
        "num_labels": 2}'


config='{"settings": {
            "model": "FacebookAI/roberta-base",
            "save_path": "'$model_path'",
            "tokenizer": "FacebookAI/roberta-base",
            "dataset":"'$dataset'", 
            "task": "glue", 
            "accelerate": true,
            "logging": false,
            "exp_name": "'$exp_name'",
            "epochs": 1,
            "batch_size":'$batch_size',  
            "lr": '$lr'},
        "config":'$model_config'}'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path

# ################################### Adaptive Baseline ###################################
batch_size=44
lr=1e-4

exp_name="adaptive_baseline"

echo Evaluating Adaptive baseline.

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

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
                "epochs": 1,
                "batch_size":'$batch_size',  
                "lr": '$lr'},
            "config":'$model_config'
        }'


python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path

################################### Sparse Baseline ###################################
batch_size=44
lr=1e-4
alpha_lr=10

exp_name="sparse_baseline"

echo Evaluating Sparse baseline.

model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

model_config='{
        "vocab_size": 32,
        "max_position_embeddings": 512,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'


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
                "epochs": 1,
                "batch_size":'$batch_size',
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path

