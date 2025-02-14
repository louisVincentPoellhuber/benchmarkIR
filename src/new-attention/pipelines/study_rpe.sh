echo Syncing...
#rsync -avz --update --progress /data/rech/poellhul/models/new-attention/ /Tmp/lvpoellhuber/models/new-attention


################################### RoBERTa ###################################
batch_size=64
lr=1e-4

# Baseline
exp_name="roberta_ape"

echo Training.

model_path=$STORAGE_DIR'/models/new-attention/roberta/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

model_config='{"max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "eager",
        "num_labels": 2
        }'

config='{"settings": {
        "task": "glue",
        "exp_name": "'$exp_name'",
        "save_path": "'$model_path'",
        "model": "FacebookAI/roberta-base",
        "train": true, 
        "validate": false, 
        "evaluate": false,
        "logging": true,
        "epochs": 10,
        "batch_size": '$batch_size',  
        "lr": '$lr'
        },
        "config":'$model_config'}'

accelerate launch src/new-attention/finetune_glue.py --config_dict "$config"
#python src/new-attention/finetune_glue.py --config_dict "$config"

# The following changes train = True to train = False, as well as 
# evaluate = False to evaluate = True. Validate is not modified as it doesn't directly impact the script. 
config=$(echo "$config" | sed -e 's/"evaluate": false/"evaluate": true/' -e 's/"train": true/"train": false/')
echo $config

# This has to be launched from Python instead of accelerate. 
python src/new-attention/finetune_glue.py --config_dict "$config"


# Baseline
exp_name="roberta_rpeK"

echo Training.

model_path=$STORAGE_DIR'/models/new-attention/roberta/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

model_config='{"max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "eager",
        "num_labels": 2, 
        "position_embedding_type": "relative_key"
        }'


config='{"settings": {
        "task": "glue",
        "exp_name": "'$exp_name'",
        "save_path": "'$model_path'",
        "model": "FacebookAI/roberta-base",
        "train": true, 
        "validate": false, 
        "evaluate": false,
        "logging": true,
        "epochs": 10,
        "batch_size": '$batch_size',  
        "lr": '$lr'
        },
        "config":'$model_config'}'

      
accelerate launch src/new-attention/finetune_glue.py --config_dict "$config"
#python src/new-attention/finetune_glue.py --config_dict "$config"

# The following changes train = True to train = False, as well as 
# evaluate = False to evaluate = True. Validate is not modified as it doesn't directly impact the script. 
config=$(echo "$config" | sed -e 's/"evaluate": false/"evaluate": true/' -e 's/"train": true/"train": false/')
echo $config

# This has to be launched from Python instead of accelerate. 
python src/new-attention/finetune_glue.py --config_dict "$config"

# Baseline
exp_name="roberta_rpeKQ"

echo Training.

model_path=$STORAGE_DIR'/models/new-attention/roberta/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

model_config='{"max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "eager",
        "num_labels": 2, 
        "position_embedding_type": "relative_key_query"
        }'


config='{"settings": {
        "task": "glue",
        "exp_name": "'$exp_name'",
        "save_path": "'$model_path'",
        "model": "FacebookAI/roberta-base",
        "train": true, 
        "validate": false, 
        "evaluate": false,
        "logging": true,
        "epochs": 10,
        "batch_size": '$batch_size',  
        "lr": '$lr'
        },
        "config":'$model_config'}'


#accelerate launch src/new-attention/finetune_glue.py --config_dict "$config"
python src/new-attention/finetune_glue.py --config_dict "$config"

# The following changes train = True to train = False, as well as 
# evaluate = False to evaluate = True. Validate is not modified as it doesn't directly impact the script. 
config=$(echo "$config" | sed -e 's/"evaluate": false/"evaluate": true/' -e 's/"train": true/"train": false/')
echo $config

# This has to be launched from Python instead of accelerate. 
python src/new-attention/finetune_glue.py --config_dict "$config"


# rsync -avz --update --progress /Tmp/lvpoellhuber/models/new-attention/ /data/rech/poellhul/models/new-attention

# scp /data/rech/poellhul/models/new-attention/experiment_df.csv ~/Downloads
