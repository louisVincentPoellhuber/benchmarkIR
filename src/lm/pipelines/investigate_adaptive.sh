echo $STORAGE_DIR

################################### Finetuned Models ###################################
batch_size=16
lr=1e-4

exp_name="finetuned"

echo Investigating finetuned models.

model_path=$STORAGE_DIR'/models/finetune_roberta/investigate_examples'
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

roberta_config='{"max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "eager",
        "num_labels": 2}'

adaptive_config='{
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

    
sparse_config='{
        "vocab_size": 32,
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'

roberta_model=$STORAGE_DIR'/models/finetune_roberta/roberta_1'
adaptive_model=$STORAGE_DIR'/models/finetune_roberta/adaptive_1'
sparse_model=$STORAGE_DIR'/models/finetune_roberta/sparse_1'


config='{"settings": {
            "roberta_model": "'$roberta_model'",
            "adaptive_model": "'$adaptive_model'",
            "sparse_model": "'$sparse_model'",
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
        "roberta_config":'$roberta_config', 
        "adaptive_config":'$adaptive_config', 
        "sparse_config":'$sparse_config'}'

python src/lm/investigate_adaptive.py --config_dict "$config"


################################### Baseline Models ###################################
batch_size=16
lr=1e-4

exp_name="baseline"

echo Investigating baseline models.

model_path=$STORAGE_DIR'/models/finetune_roberta/investigate_examples'
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

roberta_config='{"max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "eager",
        "num_labels": 2}'

adaptive_config='{
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

    
sparse_config='{
        "vocab_size": 32,
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'


config='{"settings": {
            "roberta_model": "FacebookAI/roberta-base",
            "adaptive_model": "FacebookAI/roberta-base",
            "sparse_model": "FacebookAI/roberta-base",
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
        "roberta_config":'$roberta_config', 
        "adaptive_config":'$adaptive_config', 
        "sparse_config":'$sparse_config'}'

python src/lm/investigate_adaptive.py --config_dict "$config"
