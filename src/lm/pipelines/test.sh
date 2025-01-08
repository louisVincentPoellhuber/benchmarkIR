dataset=$STORAGE_DIR'/datasets'
batch_size=32
lr=1e-4

exp_name="init_0.5_speedtest" 
dataset=$STORAGE_DIR'/datasets/qnli/qnli_train.pt'

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

# accelerate launch src/lm/finetune_roberta.py --config_dict "$config"

echo Evaluating.

#python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path --config_dict "$config"

