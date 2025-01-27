dataset=$STORAGE_DIR'/datasets'
batch_size=32
lr=1e-4


echo Roberta once more. 

exp_name="longformer_test" 

model_path=$STORAGE_DIR'/models/finetune_longformer/'$exp_name
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi


model_config='{
        "max_position_embeddings": 514,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "num_labels":4
    }'

  config='{"settings": {
          "model": "allenai/longformer-base-4096",
          "save_path": "'$model_path'",
          "tokenizer": "allenai/longformer-base-4096",
          "dataset":"'$dataset'", 
          "task": "glue", 
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
