"""
This file is for getting BERT
and RoBERTa substitutes for adjectives
in Wikipedia sentences.

Some code modified from (Bill) Yuchen Lin's example:
https://gist.github.com/yuchenlin/a2f42d3c4378ed7b83de65c7a2222eb2
"""
import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, BertForMaskedLM, RobertaForMaskedLM
from tqdm import tqdm
import time
import wikitextparser as wtp
import json
from collections import defaultdict, Counter

ROOT = '/mnt/data0/lucy/manosphere/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

def predict_masked_sent(model_name, batch_sentences, batch_idx, batch_metadata, tokenizer, top_k=20):
    if model_name == 'bert-base-uncased' or model_name == 'bert-large-uncased': 
        model = BertForMaskedLM.from_pretrained(model_name)
    elif model_name == "roberta-base": 
        model = RobertaForMaskedLM.from_pretrained(model_name)
        
    model.eval()
    model.to('cuda')
    
    with open(LOGS + 'wikipedia/substitutes/' + model_name + '.csv', 'w') as outfile: 
        for b, batch in enumerate(tqdm(batch_sentences)): 
            masked_index = batch_idx[b]
            encoded_inputs = tokenizer(batch, padding=True, truncation=True, 
                 return_tensors="pt")
            encoded_inputs.to("cuda")

            # Predict all tokens
            with torch.no_grad():
                outputs = model(**encoded_inputs)
                predictions = outputs['logits'] # [batch_size, seq_length, vocab_size]

            probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1) # [batch_size, vocab_size]
            top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True) # [batch_size, k]

            for i, pred_idx in enumerate(top_k_indices):
                predicted_tokens = tokenizer.convert_ids_to_tokens(pred_idx)
                line_ID, adj = batch_metadata[b][i]
                outfile.write(line_ID + ' ' + adj + ' ' + ' '.join(predicted_tokens) + '\n')
        
def batch_data(lines_adj, tokenizer): 
    num_lines = len(lines_adj)
    start = time.time()
    batch_size = 8
    batch_sentences = [] # list of batches
    batch_idx = [] 
    batch_metadata = [] 
    curr_batch = [] # list of list of tokens
    curr_idx = [] # list of masked_index
    curr_metadata = [] # (line_ID, adj)
    with open(LOGS + 'wikipedia/temp_adj_data/part-00000', 'r') as infile: # TODO: change to correct file
        for line in tqdm(infile, total=num_lines):
            contents = line.split('\t')
            line_num = contents[0]
            text = '\t'.join(contents[1:])
            text = wtp.remove_markup(text).lower()
            words_in_line = lines_adj[line_num]
            
            tokens = tokenizer.tokenize(text) # wordpiece tokens
            for adj in words_in_line: 
                adj_tokens = tokenizer.tokenize(adj)
                adj_len = len(adj_tokens)
                for i in range(len(tokens)): 
                    if tokens[i:i+adj_len] == adj_tokens: 
                        new_tokens = tokens[:i] + ["[MASK]"] + tokens[i+adj_len:]
                        curr_batch.append(tokenizer.convert_tokens_to_string(new_tokens))
                        curr_idx.append(i)
                        curr_metadata.append((line_num, adj))
                        if len(curr_batch) == batch_size: 
                            batch_sentences.append(curr_batch)
                            batch_idx.append(curr_idx)
                            batch_metadata.append(curr_metadata)
                            curr_batch = []
                            curr_idx = []
                            curr_metadata = []
                        break # target word found
            if len(batch_sentences) == 400: 
                break # TODO: remove
    if len(curr_batch) != 0: # fence post
        batch_sentences.append(curr_batch)
        batch_idx.append(curr_idx)
        batch_metadata.append(curr_metadata)
        
    print("TIME:", time.time() - start)
    return batch_sentences, batch_idx, batch_metadata

def predict_substitutes(model_name): 
    with open(LOGS + 'wikipedia/adj_lines.json', 'r') as infile: 
        adj_lines = json.load(infile) # {adj : [line IDs]}
        
    lines_adj = defaultdict(list) # {line ID: [adj]}
    for adj in adj_lines: 
        for line in adj_lines[adj]: 
            lines_adj[str(line)].append(adj.replace('xqxq', '-'))
            
    if model_name == 'bert-base-uncased' or model_name == 'bert-large-uncased': 
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    elif model_name == "roberta-base":
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            
    print("Getting batches...")
    batch_sentences, batch_idx, batch_metadata = batch_data(lines_adj, tokenizer)
    
    predict_masked_sent(model_name, batch_sentences, batch_idx, batch_metadata, tokenizer)
    
def find_good_contexts(model_name):
    # get mapping from each word to its synonyms and antonyms
    synonyms = defaultdict(list)
    antonyms = defaultdict(list)
    with open(LOGS + 'semantics_val/wordnet_axes.txt', 'r') as infile: 
        for line in infile: 
            contents = line.strip().split('\t') 
            if len(contents) < 3: continue # no antonyms
            synset = contents[0]
            axis1 = set(contents[1].split(','))
            axis2 = set(contents[2].split(','))
            for adj in axis1: 
                synonyms[adj] = axis1
                antonyms[adj] = axis2
            for adj in axis2: 
                synonyms[adj] = axis2
                antonyms[adj] = axis1
    
    good_line_scores = defaultdict(tuple)
    bad_line_scores = defaultdict(tuple)
    total = 0
    with open(LOGS + 'wikipedia/substitutes/' + model_name + '.csv', 'r') as infile: 
        for line in infile: 
            total += 1
            contents = line.split(' ')
            line_ID = contents[0]
            adj = contents[1]
            preds = set(contents[2:])
            ant_overlap = preds & antonyms[adj]
            if len(ant_overlap) > 0: 
                bad_line_scores[line_ID] = (adj, len(ant_overlap), ant_overlap)
                continue
            syn_overlap = preds & synonyms[adj]
            if len(syn_overlap) == 0: continue
            score = len(syn_overlap)
            good_line_scores[line_ID] = (adj, score, syn_overlap)
          
    print(len(good_line_scores), total)
    print(good_line_scores)
    print()
    print(len(bad_line_scores), total)
    print(bad_line_scores)
    print()
    
    return

    with open(LOGS + 'wikipedia/temp_adj_data/part-00000', 'r') as infile: # TODO: change to correct file
        for line in infile:
            contents = line.split('\t')
            line_num = contents[0]
            text = '\t'.join(contents[1:])
            text = wtp.remove_markup(text).lower()
            if line_num in bad_line_scores: 
                adj, score, overlap = bad_line_scores[line_num]
                print(adj, score, overlap, "-------------", text)
#             if line_num in good_line_scores: 
#                 adj, score, overlap = good_line_scores[line_num]
#                 print(adj, score, overlap, "-------------", text)

def main(): 
    predict_substitutes('bert-base-uncased')
    #predict_substitutes('bert-large-uncased')
    # TODO: roberta is broken currently due to tokenization issues
    #predict_substitutes('roberta-base')
    #find_good_contexts('bert-base-uncased')
    #find_good_contexts('bert-large-uncased')

if __name__ == '__main__':
    main()
