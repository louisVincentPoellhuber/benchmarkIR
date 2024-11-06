
################################### Adaptive ###################################
# batch_size = 44
# lr = 1e-4

echo Training Adaptive LR 5e-4.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_5e4"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_5e4",
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
                "batch_size": 44,  
                "lr": 1e-4,
                "adaptive_lr": 5e-4,
                "use_checkpoint": true, 
                "exp_name": "adaptive_lr_5e4",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_5e4"},
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

accelerate launch src/lm/evaluate_roberta.py --config_dict "$config"

echo Evaluating Adaptive 5e-4.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_5e4",
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
                "adaptive_lr": 5e-4,
                "use_checkpoint": true, 
                "exp_name": "adaptive_lr_5e4",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_5e4"},
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


echo Training Adaptive LR 1e-3.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e3"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e3",
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
                "batch_size": 44,  
                "lr": 1e-4,
                "adaptive_lr": 1e-3,
                "use_checkpoint": true, 
                "exp_name": "adaptive_lr_1e3",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e3"},
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

accelerate launch src/lm/evaluate_roberta.py --config_dict "$config"

echo Evaluating Adaptive 1e-3.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e3",
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
                "adaptive_lr": 1e-3,
                "use_checkpoint": true, 
                "exp_name": "adaptive_lr_1e3",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e3"},
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


echo Training Adaptive LR 1e-2.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e2"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e2",
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
                "batch_size": 44,  
                "lr": 1e-4,
                "adaptive_lr": 1e-2,
                "use_checkpoint": true, 
                "exp_name": "adaptive_lr_1e2",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e2"},
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

accelerate launch src/lm/evaluate_roberta.py --config_dict "$config"

echo Evaluating Adaptive 1e-2.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e2",
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
                "adaptive_lr": 1e-2,
                "use_checkpoint": true, 
                "exp_name": "adaptive_lr_1e2",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_lr_1e2"},
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



################################### Sparse 1 ###################################
# batch_size = 44
# lr = 1e-4

echo Training Sparse LR 1.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_1"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_1",
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
                "batch_size": 44,  
                "lr": 1e-4,
                "sparse_lr": 1,
                "use_checkpoint": true, 
                "exp_name": "sparse_lr_1",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_1"},
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

accelerate launch src/lm/evaluate_roberta.py --config_dict "$config"

echo Evaluating Sparse LR 1.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_1",
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
                "sparse_lr": 1,
                "use_checkpoint": true, 
                "exp_name": "sparse_lr_1",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_1"},
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


echo Training Sparse LR 2.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_2"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_2",
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
                "batch_size": 44,  
                "lr": 1e-4,
                "sparse_lr": 2,
                "use_checkpoint": true, 
                "exp_name": "sparse_lr_2",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_2"},
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

accelerate launch src/lm/evaluate_roberta.py --config_dict "$config"

echo Evaluating Sparse LR 2.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_2",
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
                "sparse_lr": 2,
                "use_checkpoint": true, 
                "exp_name": "sparse_lr_2",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_2"},
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



echo Training Sparse LR 5.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_5"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_5",
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
                "batch_size": 44,  
                "lr": 1e-4,
                "sparse_lr": 5,
                "use_checkpoint": true, 
                "exp_name": "sparse_lr_5",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_5"},
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

accelerate launch src/lm/evaluate_roberta.py --config_dict "$config"

echo Evaluating Sparse LR 5.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_5",
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
                "sparse_lr": 5,
                "use_checkpoint": true, 
                "exp_name": "sparse_lr_5",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_5"},
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



echo Training Sparse LR 15.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_15"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_15",
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
                "batch_size": 44,  
                "lr": 1e-4,
                "sparse_lr": 15,
                "use_checkpoint": true, 
                "exp_name": "sparse_lr_15",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_15"},
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

accelerate launch src/lm/evaluate_roberta.py --config_dict "$config"

echo Evaluating Sparse LR 15.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_15",
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
                "sparse_lr": 15,
                "use_checkpoint": true, 
                "exp_name": "sparse_lr_15",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_lr_15"},
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
