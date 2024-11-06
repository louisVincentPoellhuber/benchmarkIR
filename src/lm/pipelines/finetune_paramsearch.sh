################################### RoBERTa 0 ###################################
# batch_size = 44
# lr = 1e-5

# It's already been done! 
# python src/lm/metrics.py --path "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/roberta_0"


# ################################### RoBERTa 1 ###################################
# # batch_size = 44
# # lr = 1e-4

# echo Training Roberta 1.

# model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/roberta_1"
# if [[ ! -d $model_path ]]; then
#   mkdir -p $model_path
# fi

# config='{
#             "settings": {
#                 "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
#                 "model": "FacebookAI/roberta-base",
#                 "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/roberta_1",
#                 "tokenizer": "FacebookAI/roberta-base",
#                 "checkpoint": "FacebookAI/roberta-base", 
#                 "accelerate": true},
#             "preprocess_args": {
#                 "preprocess": false,
#                 "task": "glue", 
#                 "train_tokenizer": false, 
#                 "overwrite": false},
#             "train_args": {
#                 "dataset":"/part/01/Tmp/lvpoellhuber/datasets", 
#                 "epochs": 10,
#                 "batch_size": 44,  
#                 "lr": 1e-4,
#                 "use_checkpoint": true, 
#                 "exp_name": "roberta_1",
#                 "logging": true,
#                 "train": true},
#             "eval_args": {
#                 "eval": false,
#                 "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
#                 "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/roberta_1"},
#             "config": {
#                 "max_position_embeddings": 514,
#                 "hidden_size": 768,
#                 "num_attention_heads": 12,
#                 "num_hidden_layers": 6,
#                 "type_vocab_size": 1,
#                 "attn_mechanism": "eager",
#                 "num_labels": 2}
#         }'

# accelerate launch src/lm/evaluate_roberta.py --config_dict "$config"

# echo Evaluating Roberta 1.

# config='{
#             "settings": {
#                 "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
#                 "model": "FacebookAI/roberta-base",
#                 "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/roberta_1",
#                 "tokenizer": "FacebookAI/roberta-base",
#                 "checkpoint": "FacebookAI/roberta-base", 
#                 "accelerate": true},
#             "preprocess_args": {
#                 "preprocess": false,
#                 "task": "glue", 
#                 "train_tokenizer": false, 
#                 "overwrite": false},
#             "train_args": {
#                 "dataset":"/part/01/Tmp/lvpoellhuber/datasets", 
#                 "epochs": 1,
#                 "batch_size": 44,  
#                 "lr": 1e-4,
#                 "use_checkpoint": true, 
#                 "logging": false,
#                 "exp_name": "roberta_1",
#                 "train": false},
#             "eval_args": {
#                 "eval": true,
#                 "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
#                 "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/roberta_1"},
#             "config": {
#                 "max_position_embeddings": 514,
#                 "hidden_size": 768,
#                 "num_attention_heads": 12,
#                 "num_hidden_layers": 6,
#                 "type_vocab_size": 1,
#                 "attn_mechanism": "eager",
#                 "num_labels": 2}
#         }'

# python src/lm/evaluate_roberta.py --config_dict "$config"

# python src/lm/metrics.py --path $model_path

# ################################### Adaptive 0 ###################################
# # batch_size = 44
# # lr = 1e-5

# echo Training Adaptive 0.

# model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_0"
# if [[ ! -d $model_path ]]; then
# mkdir -p $model_path
# fi

# config='{
#             "settings": {
#                 "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
#                 "model": "FacebookAI/roberta-base",
#                 "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_0",
#                 "tokenizer": "FacebookAI/roberta-base",
#                 "checkpoint": "FacebookAI/roberta-base", 
#                 "accelerate": true},
#             "preprocess_args": {
#                 "preprocess": false,
#                 "task": "glue", 
#                 "train_tokenizer": false, 
#                 "overwrite": false},
#             "train_args": {
#                 "dataset":"/part/01/Tmp/lvpoellhuber/datasets", 
#                 "epochs": 10,
#                 "batch_size": 44,  
#                 "lr": 1e-5,
#                 "exp_name": "adaptive_0",
#                 "use_checkpoint": true, 
#                 "logging": true ,
#                 "train": true},
#             "eval_args": {
#                 "eval": false,
#                 "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
#                 "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_0"},
#             "config": {
#                 "max_position_embeddings": 514,
#                 "hidden_size": 768,
#                 "num_attention_heads": 12,
#                 "num_hidden_layers": 6,
#                 "type_vocab_size": 1,
#                 "attn_mechanism": "adaptive",
#                 "num_labels":4,
#                 "inner_hidden_size": 1024,
#                 "dropout": 0,
#                 "attn_span": 1024,
#                 "adapt_span_enabled": true,
#                 "adapt_span_loss": 2e-06,
#                 "adapt_span_ramp": 32,
#                 "adapt_span_init": 0,
#                 "adapt_span_cache": true
#             }
#         }'

# accelerate launch src/lm/evaluate_roberta.py --config_dict "$config"

# echo Evaluating Adaptive 0.

# config='{
#             "settings": {
#                 "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
#                 "model": "FacebookAI/roberta-base",
#                 "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_0",
#                 "tokenizer": "FacebookAI/roberta-base",
#                 "checkpoint": "FacebookAI/roberta-base", 
#                 "accelerate": true},
#             "preprocess_args": {
#                 "preprocess": false,
#                 "task": "glue", 
#                 "train_tokenizer": false, 
#                 "overwrite": false},
#             "train_args": {
#                 "dataset":"/part/01/Tmp/lvpoellhuber/datasets", 
#                 "epochs": 10,
#                 "batch_size": 32,  
#                 "lr": 1e-5,
#                 "use_checkpoint": true, 
#                 "exp_name": "adaptive_0",
#                 "logging": false,
#                 "train": false},
#             "eval_args": {
#                 "eval": true,
#                 "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
#                 "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_0"},
#             "config": {
#                 "max_position_embeddings": 514,
#                 "hidden_size": 768,
#                 "num_attention_heads": 12,
#                 "num_hidden_layers": 6,
#                 "type_vocab_size": 1,
#                 "attn_mechanism": "adaptive",
#                 "num_labels":4,
#                 "inner_hidden_size": 1024,
#                 "dropout": 0,
#                 "attn_span": 1024,
#                 "adapt_span_enabled": true,
#                 "adapt_span_loss": 2e-06,
#                 "adapt_span_ramp": 32,
#                 "adapt_span_init": 0,
#                 "adapt_span_cache": true
#             }
#         }'

# python src/lm/evaluate_roberta.py --config_dict "$config"

#python src/lm/metrics.py --path $model_path


################################### Adaptive 1 ###################################
# batch_size = 44
# lr = 1e-4

echo Training Adaptive 1.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_1"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_1",
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
                "use_checkpoint": true, 
                "exp_name": "adaptive_1",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_1"},
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

echo Evaluating Adaptive 1.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_1",
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
                "exp_name": "adaptive_1",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/adaptive_1"},
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


################################### Sparse 0 ###################################
# batch_size = 44
# lr = 1e-5

echo Training Sparse 0.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_0"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_0",
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
                "lr": 1e-5,
                "use_checkpoint": true, 
                "exp_name": "sparse_0",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_0"},
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

echo Evaluating Sparse 0.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_0",
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
                "lr": 1e-5,
                "use_checkpoint": true, 
                "exp_name": "sparse_0",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_0"},
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


################################### Sparse 1 ###################################
# batch_size = 44
# lr = 1e-4

echo Training Sparse 1.

model_path="/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_1"
if [[ ! -d $model_path ]]; then
  mkdir -p $model_path
fi

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_1",
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
                "use_checkpoint": true, 
                "exp_name": "sparse_1",
                "logging": true ,
                "train": true},
            "eval_args": {
                "eval": false,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_1"},
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

echo Evaluating Sparse 1.

config='{
            "settings": {
                "datapath":"/part/01/Tmp/lvpoellhuber/datasets", 
                "model": "FacebookAI/roberta-base",
                "save_path": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_1",
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
                "exp_name": "sparse_1",
                "logging": false,
                "train": false},
            "eval_args": {
                "eval": true,
                "dataset":"/part/01/Tmp/lvpoellhuber/datasets",
                "model": "/part/01/Tmp/lvpoellhuber/models/finetune_roberta/sparse_1"},
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

