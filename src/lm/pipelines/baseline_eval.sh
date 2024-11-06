
################################### RoBERTa Baseline ###################################
# batch_size = 44
# lr = 1e-4

echo Evaluating Roberta Baseline.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/roberta_baseline"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/roberta_baseline",
                "tokenizer": "FacebookAI/roberta-base",
                "checkpoint": "FacebookAI/roberta-base", 
                "accelerate": true},
            "preprocess_args": {
                "preprocess": false,
                "task": "glue", 
                "train_tokenizer": false, 
                "overwrite": false},
            "train_args": {
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets", 
                "epochs": 1,
                "batch_size": 44,  
                "lr": 1e-4,
                "use_checkpoint": true, 
                "logging": false,
                "exp_name": "roberta_baseline",
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "FacebookAI/roberta-base"},
            "config": {
                "max_position_embeddings": 514,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 6,
                "type_vocab_size": 1,
                "attn_mechanism": "eager",
                "num_labels": 2}
        }'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path

################################### Adaptive Baseline ###################################
#batch_size = 44
#lr = 1e-4

echo Evaluating Adaptive baseline.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_baseline"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_baseline",
                "tokenizer": "FacebookAI/roberta-base",
                "checkpoint": "FacebookAI/roberta-base", 
                "accelerate": true},
            "preprocess_args": {
                "preprocess": false,
                "task": "glue", 
                "train_tokenizer": false, 
                "overwrite": false},
            "train_args": {
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets", 
                "epochs": 10,
                "batch_size": 32,  
                "lr": 1e-4,
                "use_checkpoint": true, 
                "exp_name": "adaptive_baseline",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "FacebookAI/roberta-base"},
            "config": {
                "max_position_embeddings": 514,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 6,
                "type_vocab_size": 1,
                "attn_mechanism": "adaptive",
                "num_labels":4,
                "inner_hidden_size": 1024,
                "dropout": 0,
                "attn_span": 1024,
                "adapt_span_enabled": true,
                "adapt_span_loss": 2e-06,
                "adapt_span_ramp": 32,
                "adapt_span_init": 0,
                "adapt_span_cache": true
            }
        }'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path

################################### Sparse Baseline ###################################
# batch_size = 44
# lr = 1e-4

echo Evaluating Sparse Baseline.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_baseline"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi


config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_baseline",
                "tokenizer": "FacebookAI/roberta-base",
                "checkpoint": "FacebookAI/roberta-base", 
                "accelerate": true},
            "preprocess_args": {
                "preprocess": false,
                "task": "glue", 
                "train_tokenizer": false, 
                "overwrite": false},
            "train_args": {
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets", 
                "epochs": 10,
                "batch_size": 32,  
                "lr": 1e-4,
                "use_checkpoint": true, 
                "exp_name": "sparse_baseline",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "FacebookAI/roberta-base"},
            "config": {
                "max_position_embeddings": 514,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 6,
                "type_vocab_size": 1,
                "attn_mechanism": "sparse",
                "num_labels":4,
                "inner_hidden_size": 1024,
                "dropout": 0,
                "attn_span": 1024,
                "adapt_span_enabled": true,
                "adapt_span_loss": 2e-06,
                "adapt_span_ramp": 32,
                "adapt_span_init": 0,
                "adapt_span_cache": true
            }
        }'

python src/lm/evaluate_roberta.py --config_dict "$config"

python src/lm/metrics.py --path $model_path

