echo Syncing...
rsync -avz --update --progress /data/rech/poellhul/models/new-attention/ $STORAGE_DIR/models/new-attention


dataset=$STORAGE_DIR'/datasets'
batch_size=16
lr=1e-4

# Baseline
exp_name="scrolls_test"

echo Training.

model_path=$STORAGE_DIR'/models/new-attention/bart/'$exp_name
echo $model_path
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

dataset=$STORAGE_DIR'/datasets'

model_config='{"max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1
        }'


config='{"settings": {
        "task": "scrolls",
        "exp_name": "'$exp_name'",
        "save_path": "'$model_path'",
        "model": "facebook/bart-large",
        "tokenizer": "facebook/bart-large",
        "train": true, 
        "validate": false, 
        "evaluate": false,
        "logging": true,
        "epochs": 10,
        "train_batch_size": '$batch_size',  
        "eval_batch_size": '$batch_size',  
        "lr": '$lr'
        },
        "config":'$model_config', 
        "data": {
                "max_source_length": 1024, 
                "max_target_length": 128, 
                "pad_to_max_length": true
        }
        }'

accelerate launch src/new-attention/finetune_scrolls.py --config_dict "$config"
#python src/new-attention/finetune_glue.py --config_dict "$config"

# The following changes train = True to train = False, as well as 
# evaluate = False to evaluate = True. Validate is not modified as it doesn't directly impact the script. 
config=$(echo "$config" | sed -e 's/"evaluate": false/"evaluate": true/' -e 's/"train": true/"train": false/')
echo $config

# This has to be launched from Python instead of accelerate. 
# python src/new-attention/finetune_glue.py --config_dict "$config"

rsync -avz --update --progress $STORAGE_DIR/models/new-attention/ /data/rech/poellhul/models/new-attention
