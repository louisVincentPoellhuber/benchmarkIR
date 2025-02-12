from pathlib import Path
from tqdm import tqdm
import os
import argparse

from transformers import RobertaTokenizerFast, DataCollatorForLanguageModeling
from tokenizers import ByteLevelBPETokenizer
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import Punctuation 
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk


import dotenv
dotenv.load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

DATASETS_PATH = STORAGE_DIR+"/datasets"

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
    argparser.add_argument('--task', default="glue") # wikipedia, glue, mlm
    argparser.add_argument('--datapath', default=DATASETS_PATH) 
    argparser.add_argument('--tokenizer_path', default="FacebookAI/roberta-base")
    argparser.add_argument('--train_tokenizer', default=False) # wikipedia
    argparser.add_argument('--overwrite', default=True) # wikipedia

    args = argparser.parse_args()

    return args

def get_tokenizer(tokenizer_path):
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
    except:
        raise ValueError("Tokenizer not found. Please train a new tokenizer using src/preprocessing.py or provide a correct HF tokenizer.")
    
    return tokenizer

def get_dataloader(batch_size, dataset_path, train=True):
    dataset = Dataset(torch.load(dataset_path, weights_only=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=train)
    
    return dataloader

def get_mlm_dataloader(batch_size, dataset_path, tokenizer):
    dataset = load_from_disk(dataset_path)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True, collate_fn = data_collator.torch_call)

    return dataloader


def train_BPETokenizer(files, save_dir, vocab_size = 30522):
    tokenizer = ByteLevelBPETokenizer()

    print("Training tokenizer.")
    tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=2)
    
    print("Saving tokenizer.")
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
        sample_files_path = os.path.join(DATASETS_PATH, "wikipedia")
        if not os.path.exists(os.path.join(DATASETS_PATH, "wiki_en_0.txt")):
            dataset = load_dataset("wikipedia", language="en", date="20220301", cache_dir=DATASETS_PATH)
            create_sample_files(dataset, sample_files_path)
        paths = [str(x) for x in Path(sample_files_path).glob("*.txt")]
        return paths
    
    ''' Creates a dataset object. To do so, the function iterates through the given sample files 
    and tokenizes all the lines found inside. It then creates a Dataset object and saves it. 

    Input
        tokenizer: The tokenizer. 
        paths: The sample files to feed to the tokenizer. 
        save_path: Where to save the dataset object. 
    '''
    def generate_dataset(tokenizer, paths, save_path, overwrite=False):
        if os.path.exists(save_path) & (not overwrite):
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


def preprocess_mlm(data_path, tokenizer_path, train_tokenizer=False, overwrite=False):

    def create_sample_files(dataset, data_path, prefix):
        text_data = []
        file_count = 0

        # Create sample files for all the text data
        for sample in tqdm(dataset):
            sample = sample["text"].replace('\n', ' ')
            text_data.append(sample)
            if len(text_data) == 100000:
                file_path = os.path.join(data_path, f"{prefix}_en_{file_count}.txt")
                with open(file_path, 'w') as fp:
                    fp.write('\n'.join(text_data))
                text_data = []
                file_count += 1
        # Last sample
        with open(file_path, 'w') as fp:
            fp.write('\n'.join(text_data))

    def generate_dataset(tokenizer, paths, save_path, overwrite=False):
        if os.path.exists(save_path) & (not overwrite):
            print("Dataset already exists. ")
        else:
            print("Generating dataset object. ")

            input_ids = []
            mask = []
            batch_input_ids = None

            for path in tqdm(paths):
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.read().split("\n")
                    all_lines = " ".join(lines)
                
                tokenized_seq = tokenizer(all_lines, padding=False, return_overflowing_tokens= True, return_tensors = "pt")
            
                # Initialize the batch for the first time
                if batch_input_ids == None:
                    batch_input_ids = tokenized_seq.input_ids
                    batch_mask = tokenized_seq.attention_mask

                # If it hasn't reached 512 tokens yet, keep concatenating them
                elif batch_input_ids.shape[1]<=512:
                    batch_input_ids = torch.cat((batch_input_ids, tokenized_seq.input_ids), dim=1)
                    batch_mask = torch.cat((batch_mask, tokenized_seq.attention_mask), dim=1)

                # If it has, add the 512 tokens to the list, then keep the overflow as extra
                if batch_input_ids.shape[1]>512:
                    while batch_input_ids.shape[1]>512:
                        input_ids.append(batch_input_ids[:, :512])
                        mask.append(batch_mask[:, :512])

                        batch_input_ids = batch_input_ids[:, 512:]
                        batch_mask = batch_mask[:,512:]
                
                if len(input_ids) % 100000:
                    tensor_input_ids = torch.cat(input_ids) # concatenate all the tensors
                    tensor_mask = torch.cat(mask) 

                    encodings = {
                        "input_ids": tensor_input_ids, # tokens with mask 
                        "attention_mask": tensor_mask,
                    }

                    mlm_dataset = Dataset(encodings)

                    save_path = os.path.join(dataset_path, "mlm.pt")
                    mlm_dataset.save(save_path)
            
            tensor_input_ids = torch.cat(input_ids) # concatenate all the tensors
            tensor_mask = torch.cat(mask) 

            encodings = {
                "input_ids": tensor_input_ids, # tokens with mask 
                "attention_mask": tensor_mask,
            }

            mlm_dataset = Dataset(encodings)

            save_path = os.path.join(dataset_path, "mlm.pt")
            mlm_dataset.save(save_path)

    
    # Define the concatenation function
    def concatenate_examples(batch, tokenizer, max_length=512):
        tokenized_inputs = tokenizer(batch["text"], truncation=False, padding=False, add_special_tokens=False)
        input_ids = sum(tokenized_inputs["input_ids"], [])
        chunked_input_ids = [
            [tokenizer.cls_token_id] + input_ids[i:i + max_length - 2] + [tokenizer.sep_token_id]
            for i in range(0, len(input_ids), max_length - 2)
        ]
        chunked_input_ids = [
            chunk + [tokenizer.pad_token_id] * (max_length - len(chunk)) if len(chunk) < max_length else chunk
            for chunk in chunked_input_ids
        ]
        return {"input_ids": chunked_input_ids}
                
    # Loading / downloading datasets
    print("Loading Wikipedia.")
    wikipedia = load_dataset("wikipedia", language="en", date="20220301", cache_dir=DATASETS_PATH)
    print("Loading Book Corpus.")
    book_corpus = load_dataset("bookcorpus/bookcorpus", cache_dir=DATASETS_PATH)
    print("Loading OpenWebText.")
    openwebtext = load_dataset("Skylion007/openwebtext", cache_dir=DATASETS_PATH)
    print("Loading CC News.")
    cc_news = load_dataset("nthngdy/ccnews_split", "plain_text", cache_dir=DATASETS_PATH)

    # List them all out
    datasets = [wikipedia["train"], book_corpus["train"], openwebtext["train"], cc_news["train"], cc_news["validation"], cc_news["test"]]
    prefixes = ["wiki", "bc", "owt", "ccn", "ccn", "ccn"]

    # Create sample files for all datasets
    dataset_path = os.path.join(data_path, "mlm")
    sample_files_path = os.path.join(dataset_path, "sample_files")
    if len(os.listdir(sample_files_path))==0:
        for i, dataset in enumerate(datasets):
            print(f"Creating sample files for {prefixes[i]}.")
            create_sample_files(dataset, sample_files_path, prefixes[i])

    paths = [str(x) for x in Path(sample_files_path).glob("*.txt")]

    # Train / load the tokenizer
    if train_tokenizer:
        # Convert BPETokenizer to RobertaTokenizerFast
        # bpe_path = os.path.join(tokenizer_path, "bpe_tokenizer")
        # tokenizer = train_BPETokenizer(paths, bpe_path, 50256)
        # tokenizer = get_tokenizer(bpe_path)
        # tokenizer.save_pretrained(tokenizer_path)

        # Load FacbookeAI's RoBERTa tokenizer and save it locally
        tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = get_tokenizer(tokenizer_path)

    save_path = os.path.join(dataset_path, "mlm")
    if os.path.exists(save_path) & (not overwrite):
        print("Dataset already generated. ")
    else:
        print("Tokenizing CC News train.")
        tokenized_cc_news_train = cc_news["train"].map(
            lambda batch: concatenate_examples(batch, tokenizer),
            batched=True,
            remove_columns=["text"]
        )
        print("Tokenizing CC News test.")
        tokenized_cc_news_test = cc_news["test"].map(
            lambda batch: concatenate_examples(batch, tokenizer),
            batched=True,
            remove_columns=["text"]
        )
        print("Tokenizing CC News val.")
        tokenized_cc_news_val = cc_news["validation"].map(
            lambda batch: concatenate_examples(batch, tokenizer),
            batched=True,
            remove_columns=["text"]
        )
        print("Tokenizing wikipedia.")
        tokenized_wikipedia = wikipedia["train"].map(
            lambda batch: concatenate_examples(batch, tokenizer),
            batched=True,
            remove_columns=["text", "id", "url", "title"]
        )
        print("Tokenizing book corpus.")
        tokenized_book_corpus = book_corpus["train"].map(
            lambda batch: concatenate_examples(batch, tokenizer),
            batched=True,
            remove_columns=["text"]
        )
        print("Tokenizing openwebtext.")
        tokenized_openwebtext = openwebtext["train"].map(
            lambda batch: concatenate_examples(batch, tokenizer),
            batched=True,
            remove_columns=["text"]
        )

        print("Saving dataset.")
        full_dataset = concatenate_datasets([tokenized_wikipedia, tokenized_book_corpus, tokenized_cc_news_train, tokenized_cc_news_test, tokenized_cc_news_val, tokenized_openwebtext])
        full_dataset.save_to_disk(save_path)
        


''' This function encompasses everything needed to preprocess the MNLI dataset, from GLUE. 
Input
    data_path: Where the preprocessed data should be stored. 
    tokenizer_path: Where the tokenizer is / should be stored. 
    train_tokenizer: Whether to train the tokenizer or not. 

The function has no output, but will save the two necessary files locally: the tokenizer and the dataset. 
'''
def preprocess_glue(task, data_path, tokenizer_path, overwrite):
    
    ''' Creates a dataset object. To do so, the function iterates through the files in the eval 
    data split and tokenizes all the lines found inside. It then creates a Dataset object and saves it. 

    Input
        tokenizer: The tokenizer. 
        paths: The sample files to feed to the tokenizer. 
        save_path: Where to save the dataset object. 
    '''
    def generate_two_sentence_dataset(sentence1, sentence2, data, tokenizer, save_path, split, overwrite):
        if os.path.exists(save_path) & (not overwrite):
            print("Loading dataset. ")
            dataset = Dataset(torch.load(save_path))
        else:
            print("Generating dataset object. ")
            data_split = data[split]

            input_ids = []
            mask = []
            labels = []
            sequences = []

            for i in tqdm(range(len(data_split))):
                sequence = data_split[i][sentence1] + "</s></s>" + data_split[i][sentence2]
                tokenized_seq = tokenizer(sequence, max_length=512, padding="max_length", truncation=True, return_tensors = "pt")
                
                input_ids.append(tokenized_seq.input_ids)
                mask.append(tokenized_seq.attention_mask)
                labels.append(data_split[i]["label"])
                sequences.append(sequence)

            
            input_ids = torch.cat(input_ids) # concatenate all the tensors
            mask = torch.cat(mask) 
            labels = torch.Tensor(labels).long()

            encodings = {##
                "input_ids": input_ids, # tokens with mask 
                "attention_mask": mask,
                "labels": labels, # tokens without mask
                "sequence": sequences
            }

            dataset = Dataset(encodings)
            dataset.save(save_path)

        return dataset
    
    def generate_single_sentence_dataset(sentence, data, tokenizer, save_path, split, overwrite):
        if os.path.exists(save_path) & (not overwrite):
            print("Loading dataset. ")
            dataset = Dataset(torch.load(save_path))
        else:
            print("Generating dataset object. ")
            data_split = data[split]

            input_ids = []
            mask = []
            labels = []
            sequences = []

            for i in tqdm(range(len(data_split))):
                sequence = data_split[i][sentence]
                tokenized_seq = tokenizer(sequence, max_length=512, padding="max_length", truncation=True, return_tensors = "pt")
                
                input_ids.append(tokenized_seq.input_ids)
                mask.append(tokenized_seq.attention_mask)
                labels.append(data_split[i]["label"])
                sequences.append(sequence)

            
            input_ids = torch.cat(input_ids) # concatenate all the tensors
            mask = torch.cat(mask) 
            labels = torch.Tensor(labels).long()

            encodings = {
                "input_ids": input_ids, # tokens with mask 
                "attention_mask": mask,
                "labels": labels, # tokens without mask
                "sequence": sequences
            }

            dataset = Dataset(encodings)
            dataset.save(save_path)

        return dataset

    # Download the dataset
    data = load_dataset("glue", task, cache_dir=DATASETS_PATH)
    tokenizer = get_tokenizer(tokenizer_path)

    if task=="mnli":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("premise", "hypothesis", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("premise", "hypothesis", data, tokenizer, save_path, "validation_matched", overwrite)
    elif task=="mrpc":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "test", overwrite)
    elif (task=="stsb") | (task=="wnli"):
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "validation", overwrite)
    elif task=="rte":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("sentence1", "sentence2", data, tokenizer, save_path, "validation", overwrite)
    elif task=="qnli":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("question", "sentence", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("question", "sentence", data, tokenizer, save_path, "validation", overwrite)
    elif task=="qqp":
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_two_sentence_dataset("question1", "question2", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_two_sentence_dataset("question1", "question2", data, tokenizer, save_path, "validation", overwrite)
    elif (task=="sst2") | (task=="cola"):
        save_path = os.path.join(data_path, task+"_train.pt")
        generate_single_sentence_dataset("sentence", data, tokenizer, save_path, "train", overwrite)
        save_path = os.path.join(data_path, task+"_test.pt")
        generate_single_sentence_dataset("sentence", data, tokenizer, save_path, "validation", overwrite)


''' Main preprocessing function. Directs which preprocessing pipeline to use. 
Input
    dataset: Which dataset to preprocess. Choice between 'wikipedia', 'mnli'
    tokenizer_path: Where to save the tokenizer. 
    train_tokenizer: Whether to train the tokenizer.  
'''
def preprocess_main(task, datapath, tokenizer_path, train_tokenizer, overwrite=False):
    if task=="wikipedia":
        preprocess_wikipedia(datapath, tokenizer_path, train_tokenizer, overwrite)
    elif task=="mlm":
        preprocess_mlm(datapath, tokenizer_path, train_tokenizer, overwrite)
    elif task=="other_task":
        #raise ValueError("Invalid dataset. Please choose one of the following: ['wikipedia'].")
        pass
    elif task=="glue": # Preprocess all tasks
        tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
        #tasks = ["mnli"]
        for task in tasks:
            print(f"============ Processing {task} ============")
            task_datapath = os.path.join(datapath, task)

            if not os.path.exists(task_datapath):
                print(f"Making directory: {task_datapath}")
                os.mkdir(task_datapath)

            preprocess_glue(task, task_datapath, tokenizer_path, overwrite)
    else:
        preprocess_glue(task, datapath, tokenizer_path, overwrite)


if __name__ == "__main__":
    
    args = parse_arguments()
    
    preprocess_main(args.task, args.datapath, args.tokenizer_path, args.train_tokenizer, args.overwrite)
