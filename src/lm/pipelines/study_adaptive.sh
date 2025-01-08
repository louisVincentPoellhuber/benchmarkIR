rsync -avz --update --progress /data/rech/poellhul/models/finetune_roberta/ /Tmp/lvpoellhuber/models/finetune_roberta


################################### Adaptive ###################################
#dataset=$STORAGE_DIR'/datasets'
train_dataset=$STORAGE_DIR'/datasets/qnli/qnli_train.pt'
test_dataset=$STORAGE_DIR'/datasets/qnli/qnli_test.pt'

batch_size=32
lr=1e-4

'{
  'settings': {
    'model': 'FacebookAI/roberta-base', 
    'save_path': '/Tmp/lvpoellhuber/models/finetune_roberta/adaptive/init_0.5', 
    'tokenizer': 'FacebookAI/roberta-base', 
    'dataset': '/Tmp/lvpoellhuber/datasets', 
    'task': 'glue', 
    'accelerate': True, 
    'logging': True, 
    'exp_name': 'init_0.5', 
    'epochs': 10, 
    'batch_size': 32, 
    'lr': 0.0001}, 
  'config': {
    'max_position_embeddings': 514, 
    'hidden_size': 768, 
    'num_attention_heads': 12, 
    'num_hidden_layers': 6, 
    'type_vocab_size': 1, 
    'attn_mechanism': 'adaptive', 
    'num_labels': 4, 
    'attn_span': 1024, 
    'adapt_span_enabled': True, 
    'adapt_span_loss': 2e-06, 
    'adapt_span_ramp': 32, 
    'adapt_span_init': 0.5, 
    'adapt_span_cache': True}
  }'


echo Adaptive: No masking. 

exp_name="no_mask_test" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name'/roberta_qnli'
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
        "adapt_span_enabled": false,
        "adapt_span_loss": 2e-06, 
        "adapt_span_ramp": 32,
        "adapt_span_init": 0,
        "adapt_span_cache": true
    }'

config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$train_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.


config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Adaptive: Init @ 1, span @ 514. 

exp_name="init_5_span_514_test" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name'/roberta_qnli'
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
        "attn_span": 514,
        "adapt_span_enabled": true,
        "adapt_span_loss": 2e-06, 
        "adapt_span_ramp": 32,
        "adapt_span_init": 1,
        "adapt_span_cache": true
    }'

config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$train_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.


config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Adaptive: Init @ 0.5. 

exp_name="init_0.5_test" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name'/roberta_qnli'
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
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.


config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Adaptive: Init @ 1. 

exp_name="init_1_test" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name'/roberta_qnli'
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
        "adapt_span_init": 1,
        "adapt_span_cache": true
    }'

  config='{"settings": {
          "model": "FacebookAI/roberta-base",
          "save_path": "'$model_path'",
          "tokenizer": "FacebookAI/roberta-base",
          "dataset":"'$train_dataset'", 
          "task": "qnli", 
          "accelerate": true,
          "logging": true,
          "exp_name": "'$exp_name'",
          "epochs": 10,
          "batch_size":'$batch_size',  
          "lr": '$lr'},
      "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"


echo Evaluating.

config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Adaptive: Init @ 5. 

exp_name="init_5_test" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name'/roberta_qnli'
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
        "adapt_span_init": 5,
        "adapt_span_cache": true
    }'

  config='{"settings": {
          "model": "FacebookAI/roberta-base",
          "save_path": "'$model_path'",
          "tokenizer": "FacebookAI/roberta-base",
          "dataset":"'$dataset'", 
          "task": "qnli", 
          "accelerate": true,
          "logging": true,
          "exp_name": "'$exp_name'",
          "epochs": 10,
          "batch_size":'$batch_size',  
          "lr": '$lr'},
      "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Adaptive: Init @ 10. 

exp_name="init_10_test" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name'/roberta_qnli'
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
        "adapt_span_init": 10,
        "adapt_span_cache": true
    }'

  config='{"settings": {
          "model": "FacebookAI/roberta-base",
          "save_path": "'$model_path'",
          "tokenizer": "FacebookAI/roberta-base",
          "dataset":"'$dataset'", 
          "task": "qnli", 
          "accelerate": true,
          "logging": true,
          "exp_name": "'$exp_name'",
          "epochs": 10,
          "batch_size":'$batch_size',  
          "lr": '$lr'},
      "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Adaptive: Loss @2e-4 and init @ 0.5. 

exp_name="loss_2e-4_init_0.5_test" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name'/roberta_qnli'
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
        "adapt_span_loss": 2e-04, 
        "adapt_span_ramp": 32,
        "adapt_span_init": 0.5,
        "adapt_span_cache": true
    }'

  config='{"settings": {
          "model": "FacebookAI/roberta-base",
          "save_path": "'$model_path'",
          "tokenizer": "FacebookAI/roberta-base",
          "dataset":"'$dataset'", 
          "task": "qnli", 
          "accelerate": true,
          "logging": true,
          "exp_name": "'$exp_name'",
          "epochs": 10,
          "batch_size":'$batch_size',  
          "lr": '$lr'},
      "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Adaptive: Loss @2e-3 and init @ 0.5. 

exp_name="loss_2e-3_init_0.5_test" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name'/roberta_qnli'
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
        "adapt_span_loss": 2e-03, 
        "adapt_span_ramp": 32,
        "adapt_span_init": 0.5,
        "adapt_span_cache": true
    }'

  config='{"settings": {
          "model": "FacebookAI/roberta-base",
          "save_path": "'$model_path'",
          "tokenizer": "FacebookAI/roberta-base",
          "dataset":"'$dataset'", 
          "task": "qnli", 
          "accelerate": true,
          "logging": true,
          "exp_name": "'$exp_name'",
          "epochs": 10,
          "batch_size":'$batch_size',  
          "lr": '$lr'},
      "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

config='{"settings": {
        "model": "FacebookAI/roberta-base",
        "save_path": "'$model_path'",
        "tokenizer": "FacebookAI/roberta-base",
        "dataset":"'$test_dataset'", 
        "task": "qnli", 
        "accelerate": true,
        "logging": true,
        "exp_name": "'$exp_name'",
        "epochs": 10,
        "batch_size":'$batch_size',  
        "lr": '$lr'},
    "config":'$model_config'}'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"


# echo Adaptive: Article training strategy. 

# exp_name="adaptive_article_trainstrat" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name'/roberta_qnli'
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi


# model_config='{
#         "max_position_embeddings": 514,
#         "hidden_size": 768,
#         "num_attention_heads": 12,
#         "num_hidden_layers": 6,
#         "type_vocab_size": 1,
#         "attn_mechanism": "adaptive",
#         "num_labels":4,
#         "attn_span": 1024,
#         "adapt_span_enabled": true,
#         "adapt_span_loss": 2e-06, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "qnli", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr', 
#           "optim_strat": "constant_with_warmup", 
#           "warmup_steps": 32000
#           },
#       "config":'$model_config'}'

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



rsync -avz --update --progress /Tmp/lvpoellhuber/models/finetune_roberta/ /data/rech/poellhul/models/finetune_roberta

scp /data/rech/poellhul/models/finetune_roberta/experiment_df.csv ~/Downloads
