"""
Script for getting word embeddings 
from reddit and forum data 

One embedding for a word in each
ideology and year 

Example of use: 
python reddit_forum_embeddings.py --dataset reddit --subset 2005
python reddit_forum_embeddings.py --dataset forum --subset the_attraction
"""
from transformers import BasicTokenizer, BertTokenizerFast, BertModel, BertTokenizer
import argparse
from helpers import check_valid_comment, check_valid_post, remove_bots, get_bot_set, get_vocab
import os
import csv
from nltk import tokenize
from collections import defaultdict, Counter
import json
from tqdm import tqdm
import torch
import numpy as np

#ROOT = '/global/scratch/users/lucy3_li/manosphere/'
ROOT = '/mnt/data0/lucy/manosphere/' 
SUBS = ROOT + 'data/submissions/'
COMS = ROOT + 'data/comments/'
FORUMS = ROOT + 'data/cleaned_forums/'
ANN_FILE = ROOT + 'data/ann_sig_entities.csv'
SUB_META = ROOT + 'data/subreddits.txt'
SEM_FOLDER = ROOT + 'logs/semantics_mano/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str,
                    help='reddit, control, or forum')
parser.add_argument('--subset', type=str,
                    help='for reddit/control, should be a year, for forum, should be a forum')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_data(): 
    vocab = get_vocab()
    tokenizer = BasicTokenizer(do_lower_case=True)

    batch_size = 8
    batch_sentences = [] # each item is a list
    batch_words = [] # each item is a list
    batch_meta = []
    curr_batch = []
    curr_words = []
    curr_meta = []
    y = args.subset # somewhere between 2008 and 2019
    with open(SEM_FOLDER + args.dataset + '_' + str(y) + '_id2sent.json', 'r') as infile: 
        id2sent = json.load(infile)
    with open(SEM_FOLDER + args.dataset + '_' + str(y) + '_word2id.json', 'r') as infile: 
        word2id = json.load(infile)

    sentID_unigrams = defaultdict(list) # {sentID : [terms in line]}
    sentID_bigrams = defaultdict(list) # {sentID : [terms in line]}
    sentID_meta = defaultdict(str) # {sentID : year_category}
    for key in word2id: 
        contents = key.split('_')
        meta = '_'.join(contents[1:])
        term = contents[0]
        for sentID in word2id[key]: 
            if ' ' in term: 
                sentID_bigrams[sentID].append(term)
            else: 
                sentID_unigrams[sentID].append(term)
            sentID_meta[sentID] = meta
    
    for sentID in tqdm(id2sent): 
        sent = id2sent[sentID]
        old_tokens = tokenizer.tokenize(sent)
        meta = sentID_meta[sentID]
        unigrams = sentID_unigrams[sentID]
        if len(unigrams) > 0: 
            curr_batch.append(old_tokens)
            curr_words.append(unigrams)
            curr_meta.append(meta)
            if len(curr_batch) == batch_size: 
                batch_sentences.append(curr_batch)
                batch_words.append(curr_words)
                batch_meta.append(curr_meta)
                curr_batch = []
                curr_words = []   
                curr_meta = []
        # we treat bigrams separately in case they contain unigrams
        # this way, word ids to correspond to bigrams when needed
        bigrams = sentID_bigrams[sentID]
        if len(bigrams) > 0: 
            tokens = []
            i = 0
            while i < len(old_tokens) - 1: 
                bigram = ' '.join(old_tokens[i:i+2])
                if bigram in bigrams: 
                    tokens.append(bigram)
                    i += 2
                else: 
                    tokens.append(old_tokens[i])
                    i += 1
            curr_batch.append(tokens)
            curr_words.append(bigrams)
            curr_meta.append(meta)
            if len(curr_batch) == batch_size: 
                batch_sentences.append(curr_batch)
                batch_words.append(curr_words)
                batch_meta.append(curr_meta)
                curr_batch = []
                curr_words = []   
                curr_meta = []
    if len(curr_batch) != 0: # fence post
        batch_sentences.append(curr_batch)
        batch_words.append(curr_words)
        batch_meta.append(curr_meta)
    return batch_sentences, batch_words, batch_meta

def get_embeddings(): 
    year = args.subset
    batch_sentences, batch_words, batch_meta = batch_data()
    
    print("NUMBER OF BATCHES:", len(batch_sentences))
    
    word_reps = {}
    word_counts = Counter()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    layers = [-4, -3, -2, -1] # last four layers
    model.to(device)
    model.eval()
    
    for i, batch in enumerate(tqdm(batch_sentences)): # for every batch
        encoded_inputs = tokenizer(batch, is_split_into_words=True, padding=True, truncation=True, 
             return_tensors="pt")
        encoded_inputs.to(device)
        outputs = model(**encoded_inputs, output_hidden_states=True)
        states = outputs.hidden_states # tuple
        # batch_size x seq_len x 3072
        vector = torch.cat([states[i] for i in layers], 2) # concatenate last four
        for j in range(len(batch)): # for every example
            word_ids = encoded_inputs.word_ids(j)
            word_tokenids = defaultdict(list) # {word : [token ids]}
            for k, word_id in enumerate(word_ids): # for every token
                if word_id is not None: 
                    curr_word = batch[j][word_id]
                    if curr_word in batch_words[i][j]: 
                        word_tokenids[curr_word].append(k)
            for word in word_tokenids: 
                token_ids_word = np.array(word_tokenids[word]) 
                word_embed = vector[j][token_ids_word]
                word_embed = word_embed.mean(dim=0).detach().cpu().numpy() # average word pieces
                if np.isnan(word_embed).any(): 
                    print("PROBLEM!!!", word, batch[j])
                    return 
                word_cat = word + '_' + batch_meta[i][j]
                if word_cat not in word_reps: 
                    word_reps[word_cat] = np.zeros(3072)
                word_reps[word_cat] += word_embed
                word_counts[word_cat] += 1
        torch.cuda.empty_cache()
        
    res = {}
    for w in word_counts: 
        res[w] = list(word_reps[w] / word_counts[w])
        
    with open(SEM_FOLDER + 'embed/' + args.dataset + '_' + args.subset + '.json', 'w') as outfile: 
        json.dump(res, outfile)
        
    with open(SEM_FOLDER + 'embed/' + args.dataset + '_' + args.subset + '_wordcounts.json', 'w') as outfile: 
        json.dump(word_counts, outfile)

def main(): 
    get_embeddings()

if __name__ == '__main__':
    main()
