from roberta_config import CustomRobertaConfig
from preprocessing import get_tokenizer
from model_custom_roberta import RobertaForMaskedLM, RobertaForSequenceClassification, CustomRobertaModel
import os
from tqdm import tqdm
import json

if __name__ == "__main__":    
    model_dir = "/part/01/Tmp/lvpoellhuber/models/custom_roberta"
    tokenizer = get_tokenizer("/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_mlm")
    models = ["roberta", "adaptive", "sparse"]

    for model in models:
        model_name = model + "_mlm"
        model_path = os.path.join(model_dir, model_name)
        core_model_path = os.path.join(model_path, "roberta_model")

        if not os.path.exists(core_model_path): os.mkdir(core_model_path)

        model_type = None
        
        if model == "roberta": model = "default"

        config_name = model + "_finetune.json"
        config_path = os.path.join("/u/poellhul/Documents/Masters/benchmarkIR/src/configs", config_name)
        
        with open(config_path) as cpath:
            config = json.load(cpath)["config"]

        config = CustomRobertaConfig.from_dict(config)
        config.vocab_size = tokenizer.vocab_size+4
        
        model = RobertaForSequenceClassification(config=config).from_pretrained(model_path,config=config, ignore_mismatched_sizes=True)

        roberta_model = model.roberta
        roberta_model.save_pretrained(core_model_path, from_pt = True)
        tokenizer.save_pretrained(core_model_path)


 
        