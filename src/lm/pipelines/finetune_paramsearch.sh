rsync -avz --update --progress /data/rech/poellhul/models/finetune_roberta/ /Tmp/lvpoellhuber/models/finetune_roberta


################################### Adaptive ###################################
dataset=$STORAGE_DIR'/datasets'
batch_size=44
lr=1e-4

# exp_name="added_loss" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi


# echo Adaptive: Adding the adaptive loss term. 

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

# config='{"settings": {
#             "model": "FacebookAI/roberta-base",
#             "save_path": "'$model_path'",
#             "tokenizer": "FacebookAI/roberta-base",
#             "dataset":"'$dataset'", 
#             "task": "glue", 
#             "accelerate": true,
#             "logging": true,
#             "exp_name": "'$exp_name'",
#             "epochs": 10,
#             "batch_size":'$batch_size',  
#             "lr": '$lr'},
#         "config":'$model_config'}'


# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Increasing the adaptive loss term. 

# exp_name="loss_2e-5" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "num_labels": 4,
#         "attn_span": 1024,
#         "adapt_span_enabled": true,
#         "adapt_span_loss": 2e-05, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"


# echo Adaptive: Increasing the adaptive loss term again. 

# exp_name="loss_2e-4" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_loss": 2e-04, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"


# echo Adaptive: Increasing ramp size. 

# exp_name="ramp_64" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_ramp": 64,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"


# echo Adaptive: Decreasing ramp size. 

# exp_name="ramp_16" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_ramp": 16,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"


# echo Adaptive: Init @ 0.5. 

# exp_name="init_0.5" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_init": 0.5,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Init @ 1. 

# exp_name="init_1" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_init": 1,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"


# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Init @ 5. 

# exp_name="init_5" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_init": 5,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Init @ 10. 

# exp_name="init_10" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_init": 10,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"


# echo Adaptive: Attention span @ 1536. 

# exp_name="attn_span_1536" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "attn_span": 1536,
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
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"


# echo Adaptive: Attention span @ 2048. 

# exp_name="attn_span_2048" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "attn_span": 2048,
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
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Attention span @ 4096. 

# exp_name="attn_span_4096" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "attn_span": 4096,
#         "adapt_span_enabled": true,
#         "adapt_span_loss": 2e-06, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'

# config='{"settings": {
#         "model": "FacebookAI/roberta-base",
#         "save_path": "'$model_path'",
#         "tokenizer": "FacebookAI/roberta-base",
#         "dataset":"'$dataset'", 
#         "task": "glue", 
#         "accelerate": true,
#         "logging": true,
#         "exp_name": "'$exp_name'",
#         "epochs": 10,
#         "batch_size":'$batch_size',  
#         "lr": '$lr'},
#     "config":'$model_config'}'

# # accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config" --config "$config"




# echo Adaptive: Loss @2e-3. 

# exp_name="loss_2e-3" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_loss": 2e-03, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Loss @2e-2. 

# exp_name="loss_2e-2" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_loss": 2e-02  , 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"


# echo Adaptive: Loss @1e-1. 

# exp_name="loss_1e-1" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_loss": 1e-01, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Loss @2e-4 and init @ 0.5. 

# exp_name="loss_2e-4_init_0.5" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_loss": 2e-04, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0.5,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Loss @2e-3 and init @ 0.5. 

# exp_name="loss_2e-3_init_0.5" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_loss": 2e-03, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0.5,
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Random init. 

# exp_name="init_random" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_init": "random",
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"



# echo Adaptive: Loss @2e-4 and random init. 

# exp_name="loss_2e-4_init_random" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_loss": 2e-04, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": "random",
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"




# echo Adaptive: Loss @2e-3 and random init. 

# exp_name="loss_2e-3_init_random" 

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
#         "adapt_span_loss": 2e-03, 
#         "adapt_span_ramp": 32,
#         "adapt_span_init": "random",
#         "adapt_span_cache": true
#     }'

#   config='{"settings": {
#           "model": "FacebookAI/roberta-base",
#           "save_path": "'$model_path'",
#           "tokenizer": "FacebookAI/roberta-base",
#           "dataset":"'$dataset'", 
#           "task": "glue", 
#           "accelerate": true,
#           "logging": true,
#           "exp_name": "'$exp_name'",
#           "epochs": 10,
#           "batch_size":'$batch_size',  
#           "lr": '$lr'},
#       "config":'$model_config'}'

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"


################################### Sparse 1 ###################################


#batch_size=44
#lr=1e-4
alpha_lr=1e-1

model_config='{
        "vocab_size": 32,
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'

echo Training Sparse 1e-1. 

exp_name="sparse_1e-1"

model_path=$STORAGE_DIR'/models/finetune_roberta/sparse/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi
dataset=$STORAGE_DIR'/datasets'
config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": true,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Training Sparse LR 2e-1. 

alpha_lr=2e-1

model_config='{
        "vocab_size": 32,
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'

exp_name="sparse_2e-1"

model_path=$STORAGE_DIR'/models/finetune_roberta/sparse/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi
dataset=$STORAGE_DIR'/datasets'
config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": true,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Training Sparse LR 5e-1. 
alpha_lr=5e-1

model_config='{
        "vocab_size": 32,
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'


exp_name="sparse_5e-1"

model_path=$STORAGE_DIR'/models/finetune_roberta/sparse/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'
config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": true,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"


echo Training Sparse LR 5e-2. 
alpha_lr=5e-2

model_config='{
        "vocab_size": 32,
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'


exp_name="sparse_5e-2"

model_path=$STORAGE_DIR'/models/finetune_roberta/sparse/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'
config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": true,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



echo Training Sparse LR 2e-2. 
alpha_lr=2e-2

model_config='{
        "vocab_size": 32,
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "sparse"
    }'


exp_name="sparse_2e-2"

model_path=$STORAGE_DIR'/models/finetune_roberta/sparse/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'
config='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": "'$model_path'",
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"'$dataset'", 
                "task": "glue", 
                "accelerate": true,
                "logging": true,
                "exp_name": "'$exp_name'",
                "epochs": 10,
                "batch_size":'$batch_size',  
                "lr": '$lr',
                "alpha_lr":'$alpha_lr'},
            "config":'$model_config'
        }'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path 


rsync -avz --update --progress /Tmp/lvpoellhuber/models/finetune_roberta/ /data/rech/poellhul/models/finetune_roberta

scp /data/rech/poellhul/models/finetune_roberta/experiment_df.csv ~/Downloads
