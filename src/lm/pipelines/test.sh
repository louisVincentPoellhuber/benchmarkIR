batch_size=44
lr=1e-4

exp_name="roberta_baseline"
model_path=$STORAGE_DIR'/models/finetune_roberta/'$exp_name

echo Evaluating Roberta Baseline.

if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{"max_position_embeddings": 512,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "type_vocab_size": 1,
        "attn_mechanism": "eager",
        "num_labels": 2}'


config_dict='{
            "settings": {
                "model": "FacebookAI/roberta-base",
                "save_path": '$model_path',
                "tokenizer": "FacebookAI/roberta-base",
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets", 
                "task": "glue", 
                "accelerate": true
                "logging": false,
                "exp_name": '$exp_name',
                "epochs": 1,
                "batch_size":'$batch_size',  
                "lr": '$lr'},
            "config":'$config'
        }'
    
echo $config_dict