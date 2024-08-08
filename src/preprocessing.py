from pathlib import Path
from tqdm import tqdm
import os
import argparse

from transformers import RobertaTokenizerFast
from tokenizers import ByteLevelBPETokenizer
import torch
from datasets import load_dataset

DATASETS_PATH = "/part/01/Tmp/lvpoellhuber/datasets"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return self.encodings["input_ids"].shape[0]
    
    def __getitem__(self, i):
        return  {key: tensor[i] for key, tensor in self.encodings.items()}
    
    def save(self, save_path):
        torch.save(self.encodings, save_path)

def parse_arguments():
    argparser = argparse.ArgumentParser("masked language modeling")
    argparser.add_argument('--dataset', default="wikipedia") # wikipedia
    argparser.add_argument('--datapath', default="/part/01/Tmp/lvpoellhuber/datasets/wikipedia") 
    argparser.add_argument('--tokenizer_path', default="/part/01/Tmp/lvpoellhuber/models/custom_roberta/roberta_mlm")
    argparser.add_argument('--train_tokenizer', default=False) # wikipedia
    argparser.add_argument('--overwrite', default=False) # wikipedia

    args = argparser.parse_args()

    return args

def get_tokenizer(tokenizer_path):
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    except:
        raise ValueError("Tokenizer not found. Please train a new tokenizer using src/preprocessing.py or provide a correct HF tokenizer.")
    
    return tokenizer

def get_dataloader(batch_size, dataset_path):
    dataset = Dataset(torch.load(dataset_path))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)

    return dataloader
def train_BPETokenizer(files, save_dir):
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(files=files, vocab_size=30522, min_frequency=2, 
                    #special_tokens=[
                    #    '<s>', '<pad>', '</s>', '<unk>', '<mask>'
                    #])
    )
    
    print("Saving tokenizer...")
    tokenizer.save_model(save_dir)


def mlm(tensor):
    rand = torch.rand(tensor.shape)
    mask_arr = (rand < 0.15) * (tensor > 2)

    for i in range(tensor.shape[0]):
         selection = torch.flatten(mask_arr[i].nonzero()).tolist()
         tensor[i, selection] = 4
        
    return tensor


''' This function encompasses everything needed to preprocess the wikipedia dataset. 
Input
    data_path: Where the preprocessed data should be stored. 
    tokenizer_path: Where the tokenizer is / should be stored. 
    train_tokenizer: Whether to train the tokenizer or not. 

The function has no output, but will save the two necessary files locally: the tokenizer and the dataset. 
'''
def preprocess_wikipedia(data_path, tokenizer_path, train_tokenizer=False, overwrite=False):
    ''' Creates sample files for the wikipedia dataset. This is a way to make tokenizer training
    a bit faster. 
    Input
        dataset: The dataset object, loaded from HF. 
        out_path: Where to save dataset object.
    '''
    def create_sample_files(dataset, data_path):
        text_data = []
        file_count = 0

        print("Creating sample files.")
        # Create sample files for all the text data
        for sample in tqdm(dataset["train"]):
            sample = sample["text"].replace('\n', ' ')
            text_data.append(sample)
            if len(text_data) == 100000:
                file_path = os.path.join(data_path, f"wiki_en_{file_count}.txt")
                with open(file_path, 'w') as fp:
                    fp.write('\n'.join(text_data))
                text_data = []
                file_count += 1
        # Last sample
        with open(file_path, 'w') as fp:
            fp.write('\n'.join(text_data))

    ''' Downloads the dataset from HF or loads it up from cache. 
    Input
        data_path: Where the dataset should be stored. 
    
    Output
        paths: The different sample files' paths. 
    '''
    def download_dataset():
        # Creating sample files to simplify tokenizer training. 
        if not os.path.exists(os.path.join(DATASETS_PATH, "wiki_en_0.txt")):
            dataset = load_dataset("wikipedia", language="en", date="20220301", cache_dir=DATASETS_PATH)
            create_sample_files(dataset, DATASETS_PATH)
        paths = [str(x) for x in Path(DATASETS_PATH).glob("*.txt")]
        return paths
    
    ''' Creates a dataset object. To do so, the function iterates through the given sample files 
    and tokenizes all the lines found inside. It then creates a Dataset object and saves it. 

    Input
        tokenizer: The tokenizer. 
        paths: The sample files to feed to the tokenizer. 
        save_path: Where to save the dataset object. 
    '''
    def generate_dataset(tokenizer, paths, save_path, overwrite=False):
        if os.path.exists(save_path) & ~overwrite:
            print("Dataset already exists. ")
        else:
            print("Generating dataset object. ")

            input_ids = []
            mask = []
            labels = []

            for path in tqdm(paths):
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.read().split("\n")
                
                sample = tokenizer(lines, max_length = 512, padding = "max_length", truncation=True, return_tensors = "pt")
                labels.append(sample.input_ids)
                mask.append(sample.attention_mask)
                input_ids.append(mlm(sample.input_ids.detach().clone()))
            
            input_ids = torch.cat(input_ids) # concatenate all the tensors
            mask = torch.cat(mask) 
            labels = torch.cat(labels) 

            encodings = {
                "input_ids": input_ids, # tokens with mask 
                "attention_mask": mask,
                "labels": labels # tokens without mask
            }

            dataset = Dataset(encodings)
            dataset.save(save_path)

    paths = download_dataset()


    if train_tokenizer:
        train_BPETokenizer(paths, tokenizer_path)

    tokenizer = get_tokenizer(tokenizer_path)

    save_path = os.path.join(data_path, "dataset.pt")
    generate_dataset(tokenizer, paths, save_path, overwrite)


''' This function encompasses everything needed to preprocess the MNLI dataset, from GLUE. 
Input
    data_path: Where the preprocessed data should be stored. 
    tokenizer_path: Where the tokenizer is / should be stored. 
    train_tokenizer: Whether to train the tokenizer or not. 

The function has no output, but will save the two necessary files locally: the tokenizer and the dataset. 
'''
def preprocess_mnli(data_path, tokenizer_path, overwrite=False):
    
    ''' Creates a dataset object. To do so, the function iterates through the files in the eval 
    data split and tokenizes all the lines found inside. It then creates a Dataset object and saves it. 

    Input
        tokenizer: The tokenizer. 
        paths: The sample files to feed to the tokenizer. 
        save_path: Where to save the dataset object. 
    '''
    def generate_dataset(data, tokenizer, save_path, split, overwrite=False):
        if os.path.exists(save_path) & ~overwrite:
            print("Loading dataset. ")
            dataset = Dataset(torch.load(save_path))
        else:
            print("Generating dataset object. ")
            data_split = data[split]

            input_ids = []
            mask = []
            labels = []

            for i in tqdm(range(len(data_split))):
                sequence = data_split[i]["premise"] + ". " + data_split[i]["hypothesis"]
                tokenized_seq = tokenizer(sequence, max_length=512, padding="max_length", truncation=True, return_tensors = "pt")
                
                input_ids.append(tokenized_seq.input_ids)
                mask.append(tokenized_seq.attention_mask)
                labels.append(data_split[i]["label"])
            
            input_ids = torch.cat(input_ids) # concatenate all the tensors
            mask = torch.cat(mask) 
            labels = torch.Tensor(labels).long()

            encodings = {
                "input_ids": input_ids, # tokens with mask 
                "attention_mask": mask,
                "labels": labels # tokens without mask
            }

            dataset = Dataset(encodings)
            dataset.save(save_path)

        return dataset

    # Download the dataset
    data = load_dataset("glue", "mnli", cache_dir=DATASETS_PATH)
    tokenizer = get_tokenizer(tokenizer_path)

    save_path = os.path.join(data_path, "mnli_train.pt")
    generate_dataset(data, tokenizer, save_path, "train", overwrite)
    save_path = os.path.join(data_path, "mnli_val.pt")
    generate_dataset(data, tokenizer, save_path, "validation_matched", overwrite)
    
''' Main preprocessing function. Directs which preprocessing pipeline to use. 
Input
    dataset: Which dataset to preprocess. Choice between 'wikipedia', 'mnli'
    tokenizer_path: Where to save the tokenizer. 
    train_tokenizer: Whether to train the tokenizer.  
'''
def preprocess_main(dataset, datapath, tokenizer_path, train_tokenizer, overwrite):
    if dataset=="wikipedia":
        preprocess_wikipedia(datapath, tokenizer_path, train_tokenizer, overwrite)
    elif dataset=="mnli":
        preprocess_mnli(datapath, tokenizer_path, overwrite)
    else:
        raise ValueError("Invalid dataset. Please choose one of the following: ['wikipedia'].")

if __name__ == "__main__":
    
    args = parse_arguments()

    preprocess_main(args.dataset, args.datapath, args.tokenizer_path, args.train_tokenizer, args.overwrite)
