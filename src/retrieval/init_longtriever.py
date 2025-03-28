from transformers import AutoModel, AutoTokenizer
import argparse
from model_longtriever import Longtriever, HierarchicalLongtriever
import json
import os
import torch
import dotenv
dotenv.load_dotenv()


STORAGE_DIR = os.getenv("STORAGE_DIR")


def parse_arguments():
    argparser = argparse.ArgumentParser("BenchmarkIR Script")
    argparser.add_argument('--config', default="test") 
    argparser.add_argument('--config_dict', default={})
    argparser.add_argument('--overwrite', default=False)
    argparser.add_argument('--base_model', default="google-bert/bert-base-uncased")
    argparser.add_argument('--model_type', default="longtriever")
    
    args = argparser.parse_args()

    return args

def init_longtriever_params(base_model, longtriever):
    # Get the state_dict of the BERT encoder
    base_state_dict = base_model.state_dict()

    # Get the state_dict of the Longtriever model
    longtriever_state_dict = longtriever.state_dict()

    # Map weights from BERT to Longtriever
    new_state_dict = {}
    for bert_key, bert_value in base_state_dict.items():
        # Replace layer names to match Longtriever's naming convention
        if "layer" in bert_key:
            # Example: Map BERT's "layer.X" to Longtriever's "text_encoding_layers.X"
            new_key = bert_key.replace("layer", "text_encoding_layer")
            if new_key in longtriever_state_dict:
                new_state_dict[new_key] = bert_value

            # Example: Map BERT's "layer.X" to Longtriever's "information_exchanging.X"
            new_key = bert_key.replace("layer", "information_exchanging_layer")
            if new_key in longtriever_state_dict:
                new_state_dict[new_key] = bert_value

    # Update Longtriever's state_dict with the new weights
    longtriever_state_dict.update(new_state_dict)

    # Load the updated state_dict into Longtriever
    longtriever.load_state_dict(longtriever_state_dict)

    return longtriever

def check_weights(base_model, longtriever):
    # Compare weights
    lt_weights = longtriever.state_dict().keys()

    for name, param in base_model.named_parameters():
        name = name.replace("layer", "text_encoding_layer")
        # print(name)
        if name in lt_weights:
            lt_param = longtriever.state_dict()[name]
            if torch.equal(param, lt_param):
                print(f"Layer {name} matches")
                pass
            else:
                print(f"Layer {name} does not match")
                pass

        else:
            print(f"Layer {name} not found.")
            pass

    return

if __name__ == "__main__":
    # Parse all the arguments
    args = parse_arguments()
    
    if len(args.config_dict)>0:
        arg_dict = json.loads(args.config_dict)
    else:   
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/retrieval/configs", args.config+".json")
        with open(config_path) as fp: arg_dict = json.load(fp)
    
    for key in arg_dict["settings"]:
        if type(arg_dict["settings"][key]) == str:
            arg_dict["settings"][key] = arg_dict["settings"][key].replace("STORAGE_DIR", STORAGE_DIR)
    
    # Parameters
    config_dict = arg_dict["config"]
    model_path = args.base_model
    
    longtriever_dir = os.path.join(STORAGE_DIR, "models", args.model_type)
    
    if not os.path.exists(longtriever_dir):
        os.makedirs(longtriever_dir)

    pretrained_dir = os.path.join(longtriever_dir, "pretrained")
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
    save_path = os.path.join(pretrained_dir, model_path.split("/")[-1])

    if not args.overwrite and os.path.exists(save_path):
        print(f"Model {model_path} already exists, skipping.")
    else:
        # Load tokenizer, there should be no issues here
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        if args.model_type=="longtriever":
            model_type = Longtriever
        elif args.model_type=="hierarchical":
            model_type = HierarchicalLongtriever

        # Load the base model
        base_model = AutoModel.from_pretrained(model_path)
        # Stick the parameters that match the Longtriever model
        longtriever = model_type.from_pretrained(
                model_path,
                torch_dtype=config_dict.get("torch_dtype", "auto"),
                trust_remote_code=True,
                attn_implementation=config_dict.get("attn_implementation", "eager"),
                cache_dir=config_dict.get("cache_dir", None)
        )

        # Initialize both Longtriever encoders with the base model's encoder
        init_longtriever_params(base_model, longtriever)
        
        # Ensure the parameters are OK
        check_weights(base_model, longtriever)

        # Save everything
        longtriever.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
