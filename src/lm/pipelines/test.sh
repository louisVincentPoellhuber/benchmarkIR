#rsync -avz --update --progress /data/rech/poellhul/models/finetune_roberta/ /Tmp/lvpoellhuber/models/finetune_roberta

train_dataset=$STORAGE_DIR'/datasets/cola/cola_train.pt'
test_dataset=$STORAGE_DIR'/datasets/cola/cola_test.pt'
batch_size=32
lr=1e-4


echo Adaptive: Init @ 0.5. 

exp_name="init_0.5_newscript" 

model_path=$STORAGE_DIR'/models/finetune_roberta/test/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi


model_config='{
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "adaptive",
        "num_labels":4,
        "attn_span": 1024,
        "adapt_span_enabled": true,
        "adapt_span_loss": 2e-06, 
        "adapt_span_ramp": 32,
        "adapt_span_init": 0.5,
        "adapt_span_cache": true
    }'

config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$train_dataset'", 
        "task": "cola", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

#accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.


config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "cola", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

#python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"




echo Adaptive: Init @ 0.5. 

exp_name="init_0.5_oldscript" 

model_path=$STORAGE_DIR'/models/finetune_roberta/test/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi


model_config='{
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "adaptive",
        "num_labels":4,
        "attn_span": 1024,
        "adapt_span_enabled": true,
        "adapt_span_loss": 2e-06, 
        "adapt_span_ramp": 32,
        "adapt_span_init": 0.5,
        "adapt_span_cache": true
    }'

config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$train_dataset'", 
        "task": "cola", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

#accelerate launch src/lm/finetune_roberta_old.py --config_dict "$config"

echo Evaluating.


config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "cola", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

#python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



rsync -avz --update --progress /Tmp/lvpoellhuber/models/finetune_roberta/ /data/rech/poellhul/models/finetune_roberta

scp /data/rech/poellhul/models/finetune_roberta/experiment_df.csv ~/Downloads
