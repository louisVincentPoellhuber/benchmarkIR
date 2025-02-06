
echo Syncing...
#rsync -avz --update --progress /data/rech/poellhul/models/finetune_roberta/ /Tmp/lvpoellhuber/models/finetune_roberta


################################### RoBERTa ###################################
batch_size=32
lr=1e-4


# Baseline
exp_name="investigate_old"

echo Training.

model_path=$STORAGE_DIR'/models/finetune_roberta/roberta/'$exp_name
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
            "model": "FacebookAI/roberta-base",
            "save_path": "'$model_path'",
            "tokenizer": "FacebookAI/roberta-base",
            "dataset":"'$dataset'/cola/cola_train.pt", 
            "task": "cola", 
            "accelerate": true,
            "logging": true,
            "exp_name": "'$exp_name'",
            "epochs": 10,
            "batch_size":'$batch_size',  
            "lr": '$lr'},
        "config":'$model_config'}'

        
accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"



rsync -avz --update --progress /Tmp/lvpoellhuber/models/finetune_roberta/ /data/rech/poellhul/models/finetune_roberta

scp /data/rech/poellhul/models/finetune_roberta/experiment_df.csv ~/Downloads
