rsync -avz --update --progress /data/rech/poellhul/models/finetune_roberta/ /Tmp/lvpoellhuber/models/finetune_roberta


################################### Adaptive ###################################

# model_config='{
#         "max_position_embeddings": 514,
#         "hidden_size": 768,
#         "num_attention_heads": 12,
#         "num_hidden_layers": 6,
#         "type_vocab_size": 1,
#         "attn_mechanism": "adaptive",
#         "num_labels":4,
#         "inner_hidden_size": 1024,
#         "dropout": 0,
#         "attn_span": 1024,
#         "adapt_span_enabled": true,
#         "adapt_span_loss": 2e-06,
#         "adapt_span_ramp": 32,
#         "adapt_span_init": 0,
#         "adapt_span_cache": true
#     }'


# batch_size=44
# lr=1e-4

# exp_name="adaptive_ft"

# echo Training Adaptive 1.

# model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi

# dataset=$STORAGE_DIR'/datasets'

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

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating Adaptive 1.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"


################################### Roberta ###################################


batch_size=44
lr=1e-5

exp_name="roberta_0"

echo Finetuning Roberta 0.

model_path=$STORAGE_DIR'/models/finetune_roberta/roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

echo $model_path

dataset=$STORAGE_DIR'/datasets'

model_config='{"max_position_embeddings": 514,
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
            "logging": true,
            "exp_name": "'$exp_name'",
            "epochs": 10,
            "batch_size":'$batch_size',  
            "lr": '$lr'},
        "config":'$model_config'}'

#accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Roberta.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"


lr=1e-4

exp_name="roberta_1"

echo Finetuning Roberta 1.

model_path=$STORAGE_DIR'/models/finetune_roberta/roberta/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

echo $model_path

dataset=$STORAGE_DIR'/datasets'

model_config='{"max_position_embeddings": 514,
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
            "logging": true,
            "exp_name": "'$exp_name'",
            "epochs": 10,
            "batch_size":'$batch_size',  
            "lr": '$lr'},
        "config":'$model_config'}'

#accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating Roberta.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"

################################### Sparse ###################################

# batch_size=44
# lr=1e-4
# alpha_lr=1e-1

# model_config='{
#         "vocab_size": 32,
#         "max_position_embeddings": 514,
#         "hidden_size": 768,
#         "num_attention_heads": 12,
#         "num_hidden_layers": 6,
#         "type_vocab_size": 1,
#         "attn_mechanism": "sparse"
#     }'

# echo Training Sparse LR 1.

# exp_name="sparse_ft"

# model_path=$STORAGE_DIR'/models/finetune_roberta/sparse/'$exp_name
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi
# dataset=$STORAGE_DIR'/datasets'
# config='{
#             "settings": {
#                 "model": "FacebookAI/roberta-base",
#                 "save_path": "'$model_path'",
#                 "tokenizer": "FacebookAI/roberta-base",
#                 "dataset":"'$dataset'", 
#                 "task": "glue", 
#                 "accelerate": true,
#                 "logging": true,
#                 "exp_name": "'$exp_name'",
#                 "epochs": 10,
#                 "batch_size":'$batch_size',  
#                 "lr": '$lr',
#                 "alpha_lr":'$alpha_lr'},
#             "config":'$model_config'
#         }'

# #accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

# echo Evaluating Sparse LR 1.

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path --config_dict "$config"

rsync -avz --update --progress /Tmp/lvpoellhuber/models/finetune_roberta/ /data/rech/poellhul/models/finetune_roberta

scp /data/rech/poellhul/models/finetune_roberta/experiment_df.csv ~/Downloads