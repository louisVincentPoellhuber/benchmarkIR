rsync -avz --update --progress /data/rech/poellhul/models/finetune_roberta/ /Tmp/lvpoellhuber/models/finetune_roberta


################################### Adaptive ###################################
dataset=$STORAGE_DIR'/datasets'
batch_size=32
lr=1e-4


echo Adaptive: Init @ 0.5: speed test - before softmax.

exp_name="speedtest_before-sfmx" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
        "adapt_span_cache": true, 
        "prior_softmax": true
    }'

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
          "lr": '$lr', 
          "alpha_lr": 1e-1,
          "optim_strat": "constant_with_warmup", 
          "warmup_steps": 32000},
      "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"




echo Adaptive: Init @ 544: speed test - after softmax.

exp_name="speedtest_after-sfmx" 

model_path=$STORAGE_DIR'/models/finetune_roberta/adaptive/'$exp_name
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
        "adapt_span_cache": true, 
        "prior_softmax": false
    }'

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
          "lr": '$lr', 
          "alpha_lr": 1e-1,
          "optim_strat": "constant_with_warmup", 
          "warmup_steps": 32000},
      "config":'$model_config'}'

accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"




rsync -avz --update --progress /Tmp/lvpoellhuber/models/finetune_roberta/ /data/rech/poellhul/models/finetune_roberta

scp /data/rech/poellhul/models/finetune_roberta/experiment_df.csv ~/Downloads
